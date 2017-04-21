import os, sys
from collections import OrderedDict, namedtuple
import pandas

from . import penalties
from . import constraints
from . import backends as be

ParamsLayer = namedtuple("Params", [])

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
        # these attributes are immutable (their keys don't change)
        self.int_params = ParamsLayer()
        # these attributes are mutable (their keys do change)
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
            "layer_type"  : self.__class__.__name__,
            "intrinsic"   : list(self.int_params._fields),
            "penalties"   : {pk: self.penalties[pk].get_config()
                             for pk in self.penalties},
            "constraints" : {ck: self.constraints[ck].__name__
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
        return self.get_base_config()

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

    def save_params(self, store, key):
        """
        Save the intrinsic parameters to a HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore): the writeable stream for the params.
            key (str): the path for the layer params.

        Returns:
            None

        """
        for i, ip in enumerate(self.int_params):
            df_params = pandas.DataFrame(
                be.to_numpy_array(ip)
            )
            store.put(os.path.join(key, 'intrinsic', 'key'+str(i)), df_params)

    def load_params(self, store, key):
        """
        Load the intrinsic parameters from an HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore): the readable stream for the params.
            key (str): the path for the layer params.

        Returns:
            None

        """
        # intrinsic params
        int_params = []
        for i, ip in enumerate(self.int_params):
            int_params.append(be.float_tensor(
                store.get(os.path.join(key, 'intrinsic', 'key'+str(i))).as_matrix()
            ).squeeze()) # collapse trivial dimensions to a vector
        self.int_params = self.int_params.__class__(*int_params)

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
            self.constraints[param_name](
                getattr(self.int_params, param_name)
            )

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
            dict (float): the values of the penalty functions

        """
        pen = {param_name:
               self.penalties[param_name].value(
                   getattr(self.int_params, param_name)
               )
               for param_name in self.penalties}
        return pen

    def get_penalty_grad(self, deriv, param_name):
        """
        Get the gradient of the penalties on a parameter.

        E.g., L2 penalty gradient = penalty * parameter_i

        Args:
            deriv (tensor): derivative of the parameter
            param_name: name of the parameter

        Returns:
            tensor: derivative including penalty

        """
        if param_name not in self.penalties:
            return deriv
        else:
            return deriv + self.penalties[param_name].grad(
                getattr(self.int_params, param_name))

    def parameter_step(self, deltas):
        """
        Update the values of the intrinsic parameters:

        layer.int_params.name -= deltas.name

        Notes:
            Modifies the elements of the layer.int_params attribute in place.

        Args:
            deltas (dict): {param_name: tensor (update)}

        Returns:
            None

        """
        self.int_params = be.mapzip(be.subtract, deltas, self.int_params)
        self.enforce_constraints()


IntrinsicParamsWeights = namedtuple("IntrinsicParamsWeights", ["matrix"])

class Weights(Layer):
    """Layer class for weights"""

    def __init__(self, shape):
        """
        Create a weight layer.

        Notes:
            Simple weight layers only have a single internal parameter matrix.
            They have no external parameters because they do not depend
            on the state of anything else.

            The shape is regarded as a dimensionality of
            the visible and hidden units for the layer,
            as `shape = (visible, hidden)`.

        Args:
            shape (tuple): shape of the weight tensor (int, int)

        Returns:
            weights layer

        """
        super().__init__()
        self.shape = shape
        self.int_params = IntrinsicParamsWeights(0.01 * be.randn(shape))

    def get_config(self):
        """
        Get the configuration dictionary of the weights layer.

        Args:
            None:

        Returns:
            configuration (dict):

        """
        base_config = self.get_base_config()
        base_config["shape"] = self.shape
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create a weights layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer (Weights)

        """
        layer = cls(config["shape"])
        # TODO : params
        for k, v in config["penalties"].items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
            layer.add_constraint({k: getattr(constraints, v)})
        return layer

    def W(self):
        """
        Get the weight matrix.

        A convenience method for accessing layer.int_params.matrix
        with a shorter syntax.

        Args:
            None

        Returns:
            tensor: weight matrix

        """
        return self.int_params.matrix

    def W_T(self):
        """
        Get the transpose of the weight matrix.

        A convenience method for accessing the transpose of
        layer.int_params.matrix with a shorter syntax.

        Args:
            None

        Returns:
            tensor: transpose of weight matrix

        """
        return be.transpose(self.int_params.matrix)

    def derivatives(self, vis, hid):
        """
        Compute the derivative of the weights layer.

        dW_{ij} = - \frac{1}{num_samples} * \sum_{k} v_{ki} h_{kj}

        Args:
            vis (tensor (num_samples, num_visible)): Rescaled visible units.
            hid (tensor (num_samples, num_visible)): Rescaled hidden units.

        Returns:
            derivs (namedtuple): 'matrix': tensor (contains gradient)

        """
        derivs = IntrinsicParamsWeights(
            self.get_penalty_grad(-be.batch_outer(vis, hid) / len(vis),
                                  "matrix"))
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
        return -be.batch_dot(vis, self.W(), hid)


IntrinsicParamsGaussian = namedtuple("IntrinsicParamsGaussian", ["loc", "log_var"])
ExtrinsicParamsGaussian = namedtuple("ExtrinsicParamsGaussian", ["mean", "variance"])

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

        self.int_params = IntrinsicParamsGaussian(
            be.zeros(self.len),
            be.zeros(self.len)
        )

        self.ext_params = ExtrinsicParamsGaussian(None, None)

    def get_config(self):
        """
        Get the configuration dictionary of the Gaussian layer.

        Args:
            None:

        Returns:
            configuration (dict):

        """
        base_config = self.get_base_config()
        base_config["extrinsic"] = list(self.ext_params._fields)
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
        scale = be.exp(self.int_params.log_var)
        result = vis - be.broadcast(self.int_params.loc, vis)
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
            phi tensor (num_samples, num_units): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        scale = be.exp(self.int_params.log_var)
        logZ = be.multiply(self.int_params.loc, phi)
        logZ += be.multiply(scale, be.square(phi))
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
        # get the current values of the first and second moments
        x = self.int_params.loc
        x2 = be.exp(self.int_params.log_var) + x**2

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
        self.int_params = IntrinsicParamsGaussian(x, be.log(x2 - x**2))

    def shrink_parameters(self, shrinkage=0.1):
        """
        Apply shrinkage to the variance parameters of the layer.

        new_variance = (1-shrinkage) * old_variance + shrinkage * 1

        Notes:
            Modifies layer.int_params in place.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        var = be.exp(self.int_params.log_var)
        be.mix_inplace(be.float_scalar(1-shrinkage), var, be.ones_like(var))
        self.int_params = IntrinsicParamsGaussian(
            self.int_params.loc, be.log(var))

    def update(self, scaled_units, weights, beta=None):
        """
        Update the extrinsic parameters of the layer.

        Notes:
            Modfies layer.ext_params in place.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        mean = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            mean += be.dot(scaled_units[i], weights[i])
        if beta is not None:
            mean *= be.broadcast(beta, mean)
        mean += be.broadcast(self.int_params.loc, mean)
        var = be.broadcast(be.exp(self.int_params.log_var), mean)
        self.ext_params = ExtrinsicParamsGaussian(mean, var)

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor (num_units, num_connected_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        # initalize tensors for the location and scale derivatives
        loc = be.zeros(self.len),
        log_var = be.zeros(self.len)

        # compute the derivative with respect to the location parameter
        v_scaled = self.rescale(vis)
        loc = -be.mean(v_scaled, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')

        # compute the derivative with respect to the scale parameter
        log_var = -0.5 * be.mean(be.square(be.subtract(
            self.int_params.loc, vis)), axis=0)
        for i in range(len(hid)):
            log_var += be.batch_dot(
                hid[i],
                weights[i],
                vis,
                axis=0
            ) / len(vis)
        log_var = self.rescale(log_var)
        log_var = self.get_penalty_grad(log_var, 'log_var')

        # return the derivatives in a namedtuple
        return IntrinsicParamsGaussian(loc, log_var)

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
        scale = be.exp(self.int_params.log_var)
        return be.divide(scale, observations)

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
        return self.ext_params.mean

    def mean(self):
        """
        Compute the mean of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        return self.ext_params.mean

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        r = be.float_tensor(self.rand(be.shape(self.ext_params.mean)))
        return self.ext_params.mean + be.sqrt(self.ext_params.variance)*r

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


IntrinsicParamsIsing = namedtuple("IntrinsicParamsIsing", ["loc"])
ExtrinsicParamsIsing = namedtuple("ExtrinsicParamsIsing", ["field"])

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

        self.int_params = IntrinsicParamsIsing(be.zeros(self.len))
        self.ext_params = ExtrinsicParamsIsing(None)

    def get_config(self):
        """
        Get the configuration dictionary of the Ising layer.

        Args:
            None:

        Returns:
            configuratiom (dict):

        """
        base_config = self.get_base_config()
        base_config["extrinsic"] = list(self.ext_params._fields)
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create an Ising layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer (Ising)

        """
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        # TODO : params
        for k, v in config["penalties"].items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
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
        return -be.dot(data, self.int_params.loc)

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
        return be.logcosh(be.add(self.int_params.loc, phi))

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
        # get the current value of the first moment
        x = be.tanh(self.int_params.loc)

        # update the sample sizes
        n = len(data)
        new_sample_size = n + self.sample_size

        # updat the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the class attributes
        self.int_params = IntrinsicParamsIsing(be.atanh(x))
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the intrinsic parameters of the layer.
        Does nothing for the Ising layer.

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
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        field = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            field += be.dot(scaled_units[i], weights[i])
        if beta is not None:
            field *= be.broadcast(beta,field)
        field += be.broadcast(self.int_params.loc, field)
        self.ext_params = ExtrinsicParamsIsing(field)

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor, (num_units, num_connected_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        loc = -be.mean(vis, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')
        return IntrinsicParamsIsing(loc)

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
        return 2 * be.float_tensor(self.ext_params.field > 0) - 1

    def mean(self):
        """
        Compute the mean of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        return be.tanh(self.ext_params.field)

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        p = be.expit(self.ext_params.field)
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
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return 2 * be.float_tensor(r < 0.5) - 1


IntrinsicParamsBernoulli = namedtuple("IntrinsicParamsBernoulli", ["loc"])
ExtrinsicParamsBernoulli = namedtuple("ExtrinsicParamsBernoulli", ["field"])

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

        self.int_params = IntrinsicParamsBernoulli(be.zeros(self.len))
        self.ext_params = ExtrinsicParamsBernoulli(None)

    def get_config(self):
        """
        Get the configuration dictionary of the Bernoulli layer.

        Args:
            None:

        Returns:
            configuratiom (dict):

        """
        base_config = self.get_base_config()
        base_config["extrinsic"] = list(self.ext_params._fields)
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create a Bernoulli layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer (Bernoulli)

        """
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        # TODO : params
        for k, v in config["penalties"].items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
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
        return -be.dot(data, self.int_params.loc)

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
        return be.softplus(be.add(self.int_params.loc, phi))

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
        # get the current value of the first moment
        x = be.expit(self.int_params.loc)

        # update the sample size
        n = len(data)
        new_sample_size = n + self.sample_size

        # update the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the class attributes
        self.int_params = IntrinsicParamsBernoulli(be.logit(x))
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the intrinsic parameters of the layer.
        Does nothing for the Bernoulli layer.

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
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        field = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            field += be.dot(scaled_units[i], weights[i])
        if beta is not None:
            field *= be.broadcast(beta, field)
        field += be.broadcast(self.int_params.loc, field)
        self.ext_params = ExtrinsicParamsBernoulli(field)

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor, (num_units, num_connected_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        loc = -be.mean(vis, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')
        return IntrinsicParamsBernoulli(loc)

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
        return be.float_tensor(self.ext_params.field > 0.0)

    def mean(self):
        """
        Compute the mean of the distribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): The mean of the distribution.

        """
        return be.expit(self.ext_params.field)

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        p = be.expit(self.ext_params.field)
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
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            r = self.rand(be.shape(array_or_shape))
        except AttributeError:
            r = self.rand(array_or_shape)
        return be.float_tensor(r < 0.5)


IntrinsicParamsExponential = namedtuple("IntrinsicParamsExponential", ["loc"])
ExtrinsicParamsExponential = namedtuple("ExtrinsicParamsExponential", ["rate"])


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

        self.int_params = IntrinsicParamsExponential(be.zeros(self.len))
        self.ext_params = ExtrinsicParamsExponential(None)


    def get_config(self):
        """
        Get the configuration dictionary of the Exponential layer.

        Args:
            None:

        Returns:
            configuratiom (dict):

        """
        base_config = self.get_base_config()
        base_config["extrinsic"] = list(self.ext_params._fields)
        base_config["num_units"] = self.len
        base_config["sample_size"] = self.sample_size
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        Create an Exponential layer from a configuration dictionary.

        Args:
            config (dict)

        Returns:
            layer (Weights)

        """
        layer = cls(config["num_units"])
        layer.sample_size = config["sample_size"]
        # TODO : params
        for k, v in config["penalties"].items():
            layer.add_penalty({k: penalties.from_config(v)})
        for k, v in config["constraints"].items():
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
        return be.dot(data, self.int_params.loc)

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
        return -be.log(be.subtract(self.int_params.loc, phi))

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
        # get the current value of the first moment
        x = be.reciprocal(self.int_params.loc)

        # update the sample size
        n = len(data)
        new_sample_size = n + self.sample_size

        # update the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the class attributes
        self.int_params = IntrinsicParamsExponential(be.reciprocal(x))
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the intrinsic parameters of the layer.
        Does nothing for the Exponential layer.

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
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            None

        """
        rate = -be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            rate -= be.dot(scaled_units[i], weights[i])
        if beta is not None:
            rate *= be.broadcast(beta,rate)
        rate += be.broadcast(self.int_params.loc, rate)
        self.ext_params = ExtrinsicParamsExponential(rate)

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the intrinsic layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor, (num_units, num_connected_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        loc = be.mean(vis, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')
        return IntrinsicParamsExponential(loc)

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
        return be.reciprocal(self.ext_params.rate)

    def sample_state(self):
        """
        Draw a random sample from the disribution.

        Determined from the extrinsic parameters (layer.ext_params).

        Args:
            None

        Returns:
            tensor (num_samples, num_units): Sampled units.

        """
        r = self.rand(be.shape(self.ext_params.rate))
        return -be.log(r) / self.ext_params.rate

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For an Exponential layer, draws from the exponential distribution
        with mean 1 (i.e., Expo(1)).

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then its shape is used.

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
