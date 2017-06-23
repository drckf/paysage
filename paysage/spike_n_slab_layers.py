import os, sys
from collections import OrderedDict, namedtuple
import pandas

from .layers import Layer
from . import constraints
from . import backends as be

ParamSpikeAndSlab = namedtuple("ParamsGaussian", ["loc", "log_var"])

class SpikeAndSlabLayer(Layer):
    """Layer with Gaussian units"""

    def __init__(self, num_units):
        """
        Create a layer with Gaussian units.

        Args:
            num_units (int): the size of the layer

        Returns:
            gaussian layer

        """
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.randn
        self.params = ParamSpikeAndSlab(be.zeros(self.len), be.zeros(self.len))

    def get_config(self):
        """
        Get the configuration dictionary of the Gaussian layer.

        Args:
            None:

        Returns:
            configuration (dict):

        """
        base_config = self.get_base_config()
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create a Gaussian layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer (Gaussian)

        """
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        # TODO : params
        for k, v in config["penalties"].items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

    def energy(self, vis):
        """
        Compute the energy of the Gaussian layer.

        For sample k,
        E_k = \frac{1}{2} \sum_i \frac{(v_i - loc_i)**2}{var_i}

        Args:
            vis (tensor (num_samples, num_units)): values of units

        Returns:
            tensor (num_samples,): energy per sample

        """
        scale = be.exp(self.params.log_var)
        result = vis - be.broadcast(self.params.loc, vis)
        result = be.square(result)
        result /= be.broadcast(scale, vis)
        return 0.5 * be.mean(result, axis=1)

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let u_i and s_i be the loc and scale parameters of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = \int d x_i exp( -(x_i - u_i)^2 / (2 s_i^2) + \phi_i x_i)
        = exp(b_i u_i + b_i^2 s_i^2 / 2) sqrt(2 pi) s_i

        log(Z_i) = log(s_i) + phi_i u_i + phi_i^2 s_i^2 / 2

        Args:
            phi tensor (num_samples, num_units): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        scale = be.exp(self.params.log_var)
        logZ = be.multiply(self.params.loc, phi)
        logZ += be.multiply(scale, be.square(phi))
        logZ += be.log(be.broadcast(scale, phi))
        return logZ

    def online_param_update(self, data):
        """
        Update the parameters using an observed batch of data.
        Used for initializing the layer parameters.

        Notes:
            Modifies layer.sample_size and layer.params in place.

        Args:
            data (tensor (num_samples, num_units)): observed values for units

        Returns:
            None

        """
        # get the current values of the first and second moments
        x = self.params.loc
        x2 = be.exp(self.params.log_var) + x**2

        # update the size of the dataset
        n = len(data)
        new_sample_size = n + self.sample_size

        # update the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the second moment
        x2 *= self.sample_size / new_sample_size
        x2 += n * be.mean(be.square(data), axis=0) / new_sample_size

        # update the class attributes
        self.sample_size = new_sample_size
        self.params = ParamsGaussian(x, be.log(x2 - x**2))

    def shrink_parameters(self, shrinkage=0.1):
        """
        Apply shrinkage to the variance parameters of the layer.

        new_variance = (1-shrinkage) * old_variance + shrinkage * 1

        Notes:
            Modifies layer.params in place.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        var = be.exp(self.params.log_var)
        be.mix_inplace(be.float_scalar(1-shrinkage), var, be.ones_like(var))
        self.params = ParamsGaussian(self.params.loc, be.log(var))

    def rescale(self, observations):
        """
        Scale the observations by the variance of the layer.

        v'_i = v_i / var_i

        Args:
            observations (tensor (num_samples, num_units)):
                Values of the observed units.

        Returns:
            tensor: Rescaled observations

        """
        scale = be.exp(self.params.log_var)
        return be.divide(scale, observations)

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        # initialize tensors for the location and scale derivatives
        loc = be.zeros(self.len),
        log_var = be.zeros(self.len)

        # compute the derivative with respect to the location parameter
        v_scaled = self.rescale(vis)
        loc = -be.mean(v_scaled, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')

        # compute the derivative with respect to the scale parameter
        log_var = -0.5 * be.mean(be.square(be.subtract(
            self.params.loc, vis)), axis=0)
        for i in range(len(hid)):
            log_var += be.batch_dot(hid[i], weights[i], vis, axis=0) / len(vis)
        log_var = self.rescale(log_var)
        log_var = self.get_penalty_grad(log_var, 'log_var')

        # return the derivatives in a namedtuple
        return ParamsGaussian(loc, log_var)

    def _conditional_params(self, scaled_units, weights, beta=None):
        """
        Compute the parameters of the layer conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tuple (tensor, tensor): conditional parameters

        """
        mean = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            mean += be.dot(scaled_units[i], weights[i])
        if beta is not None:
            mean *= be.broadcast(beta, mean)
        mean += be.broadcast(self.params.loc, mean)
        var = be.broadcast(be.exp(self.params.log_var), mean)
        return mean, var

    def conditional_mode(self, scaled_units, weights, beta=None):
        """
        Compute the mode of the distribution conditioned on the state
        of the connected layers. For a Gaussian layer, the mode equals
        the mean.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor (num_samples, num_units): The mode of the distribution

        """
        mean, var = self._conditional_params(scaled_units, weights, beta)
        return mean

    def conditional_mean(self, scaled_units, weights, beta=None):
        """
        Compute the mean of the distribution conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        mean, var = self._conditional_params(scaled_units, weights, beta)
        return mean

    def conditional_sample(self, scaled_units, weights, beta=None):
        """
        Draw a random sample from the disribution conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        mean, var = self._conditional_params(scaled_units, weights, beta)
        r = be.float_tensor(self.rand(be.shape(mean)))
        return mean + be.sqrt(var)*r

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For a Gaussian layer, draws from the standard normal distribution N(0,1).

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = be.float_tensor(self.rand(be.shape(array_or_shape)))
        except AttributeError:
            r = be.float_tensor(self.rand(array_or_shape))
        return r


ParamMuSpikeAndSlab = namedtuple("ParamsGaussian", ["loc", "log_var"])

class MuSpikeAndSlabLayer(Layer):
    """Layer with Gaussian units"""

    def __init__(self, num_units):
        """
        Create a layer with Gaussian units.

        Args:
            num_units (int): the size of the layer

        Returns:
            gaussian layer

        """
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.randn
        self.params = ParamSpikeAndSlab(be.zeros(self.len), be.zeros(self.len))

    def get_config(self):
        """
        Get the configuration dictionary of the Gaussian layer.

        Args:
            None:

        Returns:
            configuration (dict):

        """
        base_config = self.get_base_config()
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create a Gaussian layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer (Gaussian)

        """
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        # TODO : params
        for k, v in config["penalties"].items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

    def energy(self, vis):
        """
        Compute the energy of the Gaussian layer.

        For sample k,
        E_k = \frac{1}{2} \sum_i \frac{(v_i - loc_i)**2}{var_i}

        Args:
            vis (tensor (num_samples, num_units)): values of units

        Returns:
            tensor (num_samples,): energy per sample

        """
        scale = be.exp(self.params.log_var)
        result = vis - be.broadcast(self.params.loc, vis)
        result = be.square(result)
        result /= be.broadcast(scale, vis)
        return 0.5 * be.mean(result, axis=1)

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let u_i and s_i be the loc and scale parameters of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = \int d x_i exp( -(x_i - u_i)^2 / (2 s_i^2) + \phi_i x_i)
        = exp(b_i u_i + b_i^2 s_i^2 / 2) sqrt(2 pi) s_i

        log(Z_i) = log(s_i) + phi_i u_i + phi_i^2 s_i^2 / 2

        Args:
            phi tensor (num_samples, num_units): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        scale = be.exp(self.params.log_var)
        logZ = be.multiply(self.params.loc, phi)
        logZ += be.multiply(scale, be.square(phi))
        logZ += be.log(be.broadcast(scale, phi))
        return logZ

    def online_param_update(self, data):
        """
        Update the parameters using an observed batch of data.
        Used for initializing the layer parameters.

        Notes:
            Modifies layer.sample_size and layer.params in place.

        Args:
            data (tensor (num_samples, num_units)): observed values for units

        Returns:
            None

        """
        # get the current values of the first and second moments
        x = self.params.loc
        x2 = be.exp(self.params.log_var) + x**2

        # update the size of the dataset
        n = len(data)
        new_sample_size = n + self.sample_size

        # update the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the second moment
        x2 *= self.sample_size / new_sample_size
        x2 += n * be.mean(be.square(data), axis=0) / new_sample_size

        # update the class attributes
        self.sample_size = new_sample_size
        self.params = ParamsGaussian(x, be.log(x2 - x**2))

    def shrink_parameters(self, shrinkage=0.1):
        """
        Apply shrinkage to the variance parameters of the layer.

        new_variance = (1-shrinkage) * old_variance + shrinkage * 1

        Notes:
            Modifies layer.params in place.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        var = be.exp(self.params.log_var)
        be.mix_inplace(be.float_scalar(1-shrinkage), var, be.ones_like(var))
        self.params = ParamsGaussian(self.params.loc, be.log(var))

    def rescale(self, observations):
        """
        Scale the observations by the variance of the layer.

        v'_i = v_i / var_i

        Args:
            observations (tensor (num_samples, num_units)):
                Values of the observed units.

        Returns:
            tensor: Rescaled observations

        """
        scale = be.exp(self.params.log_var)
        return be.divide(scale, observations)

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        # initialize tensors for the location and scale derivatives
        loc = be.zeros(self.len),
        log_var = be.zeros(self.len)

        # compute the derivative with respect to the location parameter
        v_scaled = self.rescale(vis)
        loc = -be.mean(v_scaled, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')

        # compute the derivative with respect to the scale parameter
        log_var = -0.5 * be.mean(be.square(be.subtract(
            self.params.loc, vis)), axis=0)
        for i in range(len(hid)):
            log_var += be.batch_dot(hid[i], weights[i], vis, axis=0) / len(vis)
        log_var = self.rescale(log_var)
        log_var = self.get_penalty_grad(log_var, 'log_var')

        # return the derivatives in a namedtuple
        return ParamsGaussian(loc, log_var)

    def _conditional_params(self, scaled_units, weights, beta=None):
        """
        Compute the parameters of the layer conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tuple (tensor, tensor): conditional parameters

        """
        mean = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            mean += be.dot(scaled_units[i], weights[i])
        if beta is not None:
            mean *= be.broadcast(beta, mean)
        mean += be.broadcast(self.params.loc, mean)
        var = be.broadcast(be.exp(self.params.log_var), mean)
        return mean, var

    def conditional_mode(self, scaled_units, weights, beta=None):
        """
        Compute the mode of the distribution conditioned on the state
        of the connected layers. For a Gaussian layer, the mode equals
        the mean.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor (num_samples, num_units): The mode of the distribution

        """
        mean, var = self._conditional_params(scaled_units, weights, beta)
        return mean

    def conditional_mean(self, scaled_units, weights, beta=None):
        """
        Compute the mean of the distribution conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        mean, var = self._conditional_params(scaled_units, weights, beta)
        return mean

    def conditional_sample(self, scaled_units, weights, beta=None):
        """
        Draw a random sample from the disribution conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        mean, var = self._conditional_params(scaled_units, weights, beta)
        r = be.float_tensor(self.rand(be.shape(mean)))
        return mean + be.sqrt(var)*r

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For a Gaussian layer, draws from the standard normal distribution N(0,1).

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = be.float_tensor(self.rand(be.shape(array_or_shape)))
        except AttributeError:
            r = be.float_tensor(self.rand(array_or_shape))
        return r