import sys
from collections import OrderedDict

from . import backends as be

class Layer(object):
    """A general layer class with common functionality."""
    def __init__(self, *args, **kwargs):
        """
        Basic layer initalization method.

        Args:
            *args: any arguments
            **kwargs: any keyword arguments

        Returns:
            layer

        """
        self.int_params = {}
        self.ext_params = {}
        self.penalties = OrderedDict()
        self.constraints = OrderedDict()

    def get_base_config(self):
        """
        Get a base configuration for the layer.

        Notes:
            Encodes metadata for the layer.
            Includes the base layer data.

        Args:
            None

        Returns:
            A dictionary configuration for the layer.
        """
        return {
        "layer_type": self.__class__.__name__,
        "intrinsic": list(self.int_params.keys()),
        "extrinsic": list(self.ext_params.keys()),
        "penalties": {pk: self.penalties[pk].__name__
                        for pk in self.penalties},
        "constraints": {ck: self.constraints[ck].__name__
                        for ck in self.constraints}
        }

    def get_config(self):
        """
        Get a full configuration for the layer.

        Notes:
            Encodes metadata on the layer.
            Weights are separately retrieved.
            Builds the base configuration.

        Args:
            None

        Returns:
            A dictionary configuration for the layer.
        """
        return get_base_config()

    @staticmethod
    def from_config(config):
        """
        Construct the layer from the base configuration.

        Args:
            A dictionary configuration of the layer metadata.

        Returns:
            An object which is a subclass of `Layer`.
        """
        layer_obj = getattr(sys.modules[__name__], config["layer_type"])
        return layer_obj.from_config(config)

    def add_constraint(self, constraint):
        """
        Add a parameter constraint to the layer.

        Notes:
            Modifies the layer.contraints attribute in place.

        Args:
            constraint (dict): {param_name: constraint (paysage.constraints)}

        Returns:
            None

        """
        self.constraints.update(constraint)

    def enforce_constraints(self):
        """
        Apply the contraints to the layer parameters.

        Note:
            Modifies the intrinsic parameters of the layer in place.

        Args:
            None

        Returns:
            None

        """
        for param_name in self.constraints:
            self.constraints[param_name](self.int_params[param_name])

    def add_penalty(self, penalty):
        """
        Add a penalty to the layer.

        Note:
            Modfies the layer.penalties attribute in place.

        Args:
            penalty (dict): {param_name: penalty (paysage.penalties)}

        Returns:
            None

        """
        self.penalties.update(penalty)

    def get_penalties(self):
        """
        Get the value of the penalties:

        E.g., L2 penalty = (1/2) * penalty * \sum_i parameter_i ** 2

        Args:
            None

        Returns:
            float: the value of the penalty functions

        """
        pen = {param_name:
            self.penalties[param_name].value(self.int_params[param_name])
            for param_name in self.penalties}
        return pen

    def get_penalty_gradients(self):
        """
        Get the gradients of the penalties.

        E.g., L2 penalty = penalty * parameter_i

        Args:
            None

        Returns:
            pen (dict): {param_name: tensor (containing gradient)}

        """
        pen = {param_name:
            self.penalties[param_name].grad(self.int_params[param_name])
            for param_name in self.penalties}
        return pen

    def parameter_step(self, deltas):
        """
        Update the values of the intrinsic parameters:

        layer.int_params['name'] -= deltas['name']

        Notes:
            Modifies the layer.int_params attribute in place.

        Args:
            deltas (dict): {param_name: tensor (update)}

        Returns:
            None

        """
        be.subtract_dicts_inplace(self.int_params, deltas)
        self.enforce_constraints()


class Weights(Layer):
    """Layer class for weights"""
    def __init__(self, shape):
        """
        Create a weight layer.

        Notes:
            Simple weight layers only have a single internal parameter matrix.
            They have no external parameters because they do not depend
            on the state of anything else.

        Args:
            shape (tuple): shape of the weight tensor (int, int)

        Returns:
            weights layer

        """
        super().__init__()

        self.shape = shape

        self.int_params = {
        'matrix': 0.01 * be.randn(shape)
        }

    def get_config(self):
        base_config = self.get_base_config()
        base_config["shape"] = self.shape
        return base_config

    @classmethod
    def from_config(cls, config):
        layer = cls(config["shape"])
        for k, v in config["penalties"]:
            layer.add_penalty({k: getattr(penalties, v)})
        for k, v in config["constraints"]:
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

    def W(self):
        """
        Get the weight matrix.

        A convenience method for accessing layer.int_params['matrix']
        with a shorter syntax.

        Args:
            None

        Returns:
            tensor: weight matrix

        """
        return self.int_params['matrix']

    def W_T(self):
        """
        Get the transpose of the weight matrix.

        A convenience method for accessing the transpose of
        layer.int_params['matrix'] with a shorter syntax.

        Args:
            None

        Returns:
            tensor: transpose of weight matrix

        """
        return be.transpose(self.int_params['matrix'])

    def derivatives(self, vis, hid):
        """
        Compute the derivative of the weights layer.

        dW_{ij} = - \frac{1}{num_samples} * \sum_{k} v_{ki} h_{kj}

        Args:
            vis (tensor (num_samples, num_visible)): Rescaled visible units.
            hid (tensor (num_samples, num_visible)): Rescaled hidden units.

        Returns:
            derivs (dict): {'matrix': tensor (contains gradient)}

        """
        n = len(vis)
        derivs = {
        'matrix': -be.batch_outer(vis, hid) / n
        }
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())
        return derivs

    def energy(self, vis, hid):
        """
        Compute the contribution of the weight layer to the model energy.

        For sample k:
        E_k = -\sum_{ij} W_{ij} v_{ki} h_{kj}

        Args:
            vis (tensor (num_samples, num_visible)): Rescaled visible units.
            hid (tensor (num_samples, num_visible)): Rescaled hidden units.

        Returns:
            tensor (num_samples,): energy per sample

        """
        return -be.batch_dot(vis, self.int_params['matrix'], hid)


class GaussianLayer(Layer):
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

        self.int_params = {
        'loc': be.zeros(self.len),
        'log_var': be.zeros(self.len)
        }

        self.ext_params = {
        'mean': None,
        'variance': None
        }

    def get_config(self):
        base_config = self.get_base_config()
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        for k, v in config["penalties"]:
            layer.add_penalty({k: getattr(penalties, v)})
        for k, v in config["constraints"]:
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
        scale = be.exp(self.int_params['log_var'])
        result = vis - be.broadcast(self.int_params['loc'], vis)
        result = be.square(result)
        result /= be.broadcast(scale, vis)
        return 0.5 * be.mean(result, axis=1)

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let u_i and s_i be the intrinsic loc and scale parameters of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = \int d x_i exp( -(x_i - u_i)^2 / (2 s_i^2) + \phi_i x_i)
        = exp(b_i u_i + b_i^2 s_i^2 / 2) sqrt(2 pi) s_i

        log(Z_i) = log(s_i) + phi_i u_i + phi_i^2 s_i^2 / 2

        Args:
            phi (tensor (num_samples, num_units)): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        scale = be.exp(self.int_params['log_var'])

        logZ = be.broadcast(self.int_params['loc'], phi) * phi
        logZ += be.broadcast(scale, phi) * be.square(phi)
        logZ += be.log(be.broadcast(scale, phi))

        return logZ

    def online_param_update(self, data):
        """
        Update the intrinsic parameters using an observed batch of data.
        Used for initializing the layer parameters.

        Notes:
            Modifies layer.sample_size and layer.int_params in place.

        Args:
            data (tensor (num_samples, num_units)): observed values for units

        Returns:
            None

        """
        n = len(data)
        new_sample_size = n + self.sample_size
        # compute the current value of the second moment
        x2 = be.exp(self.int_params['log_var'])
        x2 += self.int_params['loc']**2
        # update the first moment / location parameter
        self.int_params['loc'] *= self.sample_size / new_sample_size
        self.int_params['loc'] += n * be.mean(data, axis=0) / new_sample_size
        # update the second moment
        x2 *= self.sample_size / new_sample_size
        x2 += n * be.mean(be.square(data), axis=0) / new_sample_size
        # update the log_var parameter from the second moment
        self.int_params['log_var'] = be.log(x2 - self.int_params['loc']**2)
        # update the sample size
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=0.1):
        """
        Apply shrinkage to the variance parameters of the layer.

        new_variance = (1-shrinkage) * old_variance + shrinkage * 1

        Notes:
            Modifies layer.int_params['loc_var'] in place.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        var = be.exp(self.int_params['log_var'])
        be.mix_inplace(be.float_scalar(1-shrinkage), var, be.ones_like(var))
        self.int_params['log_var'] = be.log(var)

    def update(self, scaled_units, weights, beta=None):
        """
        Update the extrinsic parameters of the layer.

        Notes:
            Modfies layer.ext_params in place.

        Args:
            scaled_units (tensor (num_samples, num_connected_units)):
                The rescaled values of the connected units.
            weights (tensor, (num_connected_units, num_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        self.ext_params['mean'] = be.dot(scaled_units, weights)
        if beta is not None:
            self.ext_params['mean'] *= be.broadcast(
                                       beta,
                                       self.ext_params['mean']
                                       )
        self.ext_params['mean'] += be.broadcast(
                                   self.int_params['loc'],
                                   self.ext_params['mean']
                                   )
        self.ext_params['variance'] = be.broadcast(
                                      be.exp(self.int_params['log_var']),
                                      self.ext_params['mean']
                                      )

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid (tensor (num_samples, num_connected_units)):
                The rescaled values of the hidden units.
            weights (tensor, (num_units, num_connected_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (dict): {param_name: tensor (contains gradient)}

        """
        derivs = {
        'loc': be.zeros(self.len),
        'log_var': be.zeros(self.len)
        }

        v_scaled = self.rescale(vis)
        derivs['loc'] = -be.mean(v_scaled, axis=0)

        diff = be.square(
        vis - be.broadcast(self.int_params['loc'], vis)
        )
        derivs['log_var'] = -0.5 * be.mean(diff, axis=0)
        derivs['log_var'] += be.batch_dot(
                             hid,
                             be.transpose(weights),
                             vis,
                             axis=0
                             ) / len(vis)
        derivs['log_var'] = self.rescale(derivs['log_var'])

        be.add_dicts_inplace(derivs, self.get_penalty_gradients())
        return derivs

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
        scale = be.exp(self.int_params['log_var'])
        return observations / be.broadcast(scale, observations)

    def mode(self):
        """
        Compute the mode of the distribution.
        For a Gaussian layer, the mode equals the mean.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mode of the distribution

        """
        return self.ext_params['mean']

    def mean(self):
        """
        Compute the mean of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        return self.ext_params['mean']

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        r = be.float_tensor(self.rand(be.shape(self.ext_params['mean'])))
        return self.ext_params['mean'] + be.sqrt(self.ext_params['variance'])*r

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For a Gaussian layer, draws from the standard normal distribution N(0,1).

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then it's shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = be.float_tensor(self.rand(be.shape(array_or_shape)))
        except AttributeError:
            r = be.float_tensor(self.rand(array_or_shape))
        return r


class IsingLayer(Layer):
    """Layer with Ising units (i.e., -1 or +1)."""
    def __init__(self, num_units):
        """
        Create a layer with Ising units.

        Args:
            num_units (int): the size of the layer

        Returns:
            ising layer

        """
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.rand

        self.int_params = {
        'loc': be.zeros(self.len)
        }

        self.ext_params = {
        'field': None
        }

    def get_config(self):
        base_config = self.get_base_config()
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        for k, v in config["penalties"]:
            layer.add_penalty({k: getattr(penalties, v)})
        for k, v in config["constraints"]:
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

    def energy(self, data):
        """
        Compute the energy of the Ising layer.

        For sample k,
        E_k = -\sum_i loc_i * v_i

        Args:
            vis (tensor (num_samples, num_units)): values of units

        Returns:
            tensor (num_samples,): energy per sample

        """
        return -be.dot(data, self.int_params['loc'])

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let a_i be the intrinsic loc parameter of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = Tr_{x_i} exp( a_i x_i + phi_i x_i)
        = 2 cosh(a_i + phi_i)

        log(Z_i) = logcosh(a_i + phi_i)

        Args:
            phi (tensor (num_samples, num_units)): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        logZ = be.broadcast(self.int_params['loc'], phi) + phi
        return be.logcosh(logZ)

    def online_param_update(self, data):
        """
        Update the intrinsic parameters using an observed batch of data.
        Used for initializing the layer parameters.

        Notes:
            Modifies layer.sample_size and layer.int_params in place.

        Args:
            data (tensor (num_samples, num_units)): observed values for units

        Returns:
            None

        """
        n = len(data)
        new_sample_size = n + self.sample_size
        # update the first moment
        x = be.tanh(self.int_params['loc'])
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size
        # update the location parameter
        self.int_params['loc'] = be.atanh(x)
        # update the sample size
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the intrinsic parameters of the layer.
        Does nothing for the Ising layer.

        Notes:
            Modifies layer.int_params['loc_var'] in place.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        pass

    def update(self, scaled_units, weights, beta=None):
        """
        Update the extrinsic parameters of the layer.

        Notes:
            Modfies layer.ext_params in place.

        Args:
            scaled_units (tensor (num_samples, num_connected_units)):
                The rescaled values of the connected units.
            weights (tensor, (num_connected_units, num_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        self.ext_params['field'] = be.dot(scaled_units, weights)
        if beta is not None:
            self.ext_params['field'] *= be.broadcast(
                                        beta,
                                        self.ext_params['field']
                                        )
        self.ext_params['field'] += be.broadcast(
                                    self.int_params['loc'],
                                    self.ext_params['field']
                                    )

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid (tensor (num_samples, num_connected_units)):
                The rescaled values of the hidden units.
            weights (tensor, (num_units, num_connected_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (dict): {param_name: tensor (contains gradient)}

        """
        derivs = {
        'loc': be.zeros(self.len)
        }

        derivs['loc'] = -be.mean(vis, axis=0)
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())

        return derivs

    def rescale(self, observations):
        """
        Rescale is equivalent to the identity function for the Ising layer.

        Args:
            observations (tensor (num_samples, num_units)):
                Values of the observed units.

        Returns:
            tensor: observations

        """
        return observations

    def mode(self):
        """
        Compute the mode of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mode of the distribution

        """
        return 2 * be.float_tensor(self.ext_params['field'] > 0) - 1

    def mean(self):
        """
        Compute the mean of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        return be.tanh(self.ext_params['field'])

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        p = be.expit(self.ext_params['field'])
        r = self.rand(be.shape(p))
        return 2 * be.float_tensor(r < p) - 1

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For an Ising layer, draws -1 or +1 with equal probablity.

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then it's shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return 2 * be.float_tensor(r < 0.5) - 1


class BernoulliLayer(Layer):
    """Layer with Bernoulli units (i.e., 0 or +1)."""
    def __init__(self, num_units):
        """
        Create a layer with Bernoulli units.

        Args:
            num_units (int): the size of the layer

        Returns:
            bernoulli layer

        """
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.rand

        self.int_params = {
        'loc': be.zeros(self.len)
        }

        self.ext_params = {
        'field': None
        }

    def get_config(self):
        base_config = self.get_base_config()
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        for k, v in config["penalties"]:
            layer.add_penalty({k: getattr(penalties, v)})
        for k, v in config["constraints"]:
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

    def energy(self, data):
        """
        Compute the energy of the Bernoulli layer.

        For sample k,
        E_k = -\sum_i loc_i * v_i

        Args:
            vis (tensor (num_samples, num_units)): values of units

        Returns:
            tensor (num_samples,): energy per sample

        """
        return -be.dot(data, self.int_params['loc'])

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let a_i be the intrinsic loc parameter of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = Tr_{x_i} exp( a_i x_i + phi_i x_i)
        = 1 + exp(a_i + phi_i)

        log(Z_i) = softplus(a_i + phi_i)

        Args:
            phi (tensor (num_samples, num_units)): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        logZ = be.broadcast(self.int_params['loc'], phi) + phi
        return be.softplus(logZ)

    def online_param_update(self, data):
        """
        Update the intrinsic parameters using an observed batch of data.
        Used for initializing the layer parameters.

        Notes:
            Modifies layer.sample_size and layer.int_params in place.

        Args:
            data (tensor (num_samples, num_units)): observed values for units

        Returns:
            None

        """
        n = len(data)
        new_sample_size = n + self.sample_size
        # update the first moment
        x = be.expit(self.int_params['loc'])
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size
        # update the location parameter
        self.int_params['loc'] = be.logit(x)
        # update the sample size
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the intrinsic parameters of the layer.
        Does nothing for the Bernoulli layer.

        Notes:
            Modifies layer.int_params['loc_var'] in place.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        pass

    def update(self, scaled_units, weights, beta=None):
        """
        Update the extrinsic parameters of the layer.

        Notes:
            Modfies layer.ext_params in place.

        Args:
            scaled_units (tensor (num_samples, num_connected_units)):
                The rescaled values of the connected units.
            weights (tensor, (num_connected_units, num_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        self.ext_params['field'] = be.dot(scaled_units, weights)
        if beta is not None:
            self.ext_params['field'] *= be.broadcast(
                                        beta,
                                        self.ext_params['field']
                                        )
        self.ext_params['field'] += be.broadcast(
                                    self.int_params['loc'],
                                    self.ext_params['field']
                                    )

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid (tensor (num_samples, num_connected_units)):
                The rescaled values of the hidden units.
            weights (tensor, (num_units, num_connected_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (dict): {param_name: tensor (contains gradient)}

        """
        derivs = {
        'loc': be.zeros(self.len)
        }

        derivs['loc'] = -be.mean(vis, axis=0)
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())

        return derivs

    def rescale(self, observations):
        """
        Rescale is equivalent to the identity function for the Bernoulli layer.

        Args:
            observations (tensor (num_samples, num_units)):
                Values of the observed units.

        Returns:
            tensor: observations

        """
        return observations

    def mode(self):
        """
        Compute the mode of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mode of the distribution

        """
        return be.float_tensor(self.ext_params['field'] > 0.0)

    def mean(self):
        """
        Compute the mean of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        return be.expit(self.ext_params['field'])

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        p = be.expit(self.ext_params['field'])
        r = self.rand(be.shape(p))
        return be.float_tensor(r < p)

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For a Bernoulli layer, draws 0 or 1 with equal probability.

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then it's shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return be.float_tensor(r < 0.5)


class ExponentialLayer(Layer):
    """Layer with Exponential units (non-negative)."""
    def __init__(self, num_units):
        """
        Create a layer with Exponential units.

        Args:
            num_units (int): the size of the layer

        Returns:
            exponential layer

        """
        super().__init__()

        self.len = num_units
        self.sample_size = 0
        self.rand = be.rand

        self.int_params = {
        'loc': be.zeros(self.len)
        }

        self.ext_params = {
        'rate': None
        }

    def get_config(self):
        base_config = self.get_base_config()
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        for k, v in config["penalties"]:
            layer.add_penalty({k: getattr(penalties, v)})
        for k, v in config["constraints"]:
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

    def energy(self, data):
        """
        Compute the energy of the Exponential layer.

        For sample k,
        E_k = \sum_i loc_i * v_i

        Args:
            vis (tensor (num_samples, num_units)): values of units

        Returns:
            tensor (num_samples,): energy per sample

        """
        return be.dot(data, self.int_params['loc'])

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let a_i be the intrinsic loc parameter of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = Tr_{x_i} exp( -a_i x_i + phi_i x_i)
        = 1 / (a_i - phi_i)

        log(Z_i) = -log(a_i - phi_i)

        Args:
            phi (tensor (num_samples, num_units)): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        logZ = be.broadcast(self.int_params['loc'], phi) - phi
        return -be.log(logZ)

    def online_param_update(self, data):
        """
        Update the intrinsic parameters using an observed batch of data.
        Used for initializing the layer parameters.

        Notes:
            Modifies layer.sample_size and layer.int_params in place.

        Args:
            data (tensor (num_samples, num_units)): observed values for units

        Returns:
            None

        """
        n = len(data)
        new_sample_size = n + self.sample_size
        # update the first moment
        x = self.mean(self.int_params['loc'])
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size
        # update the location parameter
        self.int_params['loc'] = be.reciprocal(x)
        # update the sample size
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the intrinsic parameters of the layer.
        Does nothing for the Exponential layer.

        Notes:
            Modifies layer.int_params['loc_var'] in place.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        pass

    def update(self, scaled_units, weights, beta=None):
        """
        Update the extrinsic parameters of the layer.

        Notes:
            Modfies layer.ext_params in place.

        Args:
            scaled_units (tensor (num_samples, num_connected_units)):
                The rescaled values of the connected units.
            weights (tensor, (num_connected_units, num_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        self.ext_params['rate'] = -be.dot(scaled_units, weights)
        if beta is not None:
            self.ext_params['rate'] *= be.broadcast(
                                        beta,
                                        self.ext_params['rate']
                                        )
        self.ext_params['rate'] += be.broadcast(
                                    self.int_params['loc'],
                                    self.ext_params['rate']
                                    )

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid (tensor (num_samples, num_connected_units)):
                The rescaled values of the hidden units.
            weights (tensor, (num_units, num_connected_units)):
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (dict): {param_name: tensor (contains gradient)}

        """
        derivs = {
        'loc': be.zeros(self.len)
        }

        derivs['loc'] = be.mean(vis, axis=0)
        be.add_dicts_inplace(derivs, self.get_penalty_gradients())

        return derivs

    def rescale(self, observations):
        """
        Rescale is equivalent to the identity function for the Exponential layer.

        Args:
            observations (tensor (num_samples, num_units)):
                Values of the observed units.

        Returns:
            tensor: observations

        """
        return observations

    def mode(self):
        """
        The mode of the Exponential distribution is undefined.

        Args:
            None

        Raises:
            NotImplementedError

        """
        raise NotImplementedError("Exponential distribution has no mode.")

    def mean(self):
        """
        Compute the mean of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        return be.reciprocal(self.ext_params['rate'])

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        r = self.rand(be.shape(self.ext_params['rate']))
        return -be.log(r) / self.ext_params['rate']

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For an Exponential layer, draws from the exponential distribution
        with mean 1 (i.e., Expo(1)).

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then it's shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return -be.log(r)


# ---- FUNCTIONS ----- #

def get(key):
    if 'gauss' in key.lower():
        return GaussianLayer
    elif 'ising' in key.lower():
        return IsingLayer
    elif 'bern' in key.lower():
        return BernoulliLayer
    elif 'expo' in key.lower():
        return ExponentialLayer
    else:
        raise ValueError('Unknown layer type')
