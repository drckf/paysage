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
        Basic layer initialization method.

        Args:
            *args: any arguments
            **kwargs: any keyword arguments

        Returns:
            layer

        """
        # these attributes are immutable (their keys don't change)
        self.params = ParamsLayer()
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
            "parameters"  : list(self.params._fields),
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
        Save the parameters to a HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore): the writeable stream for the params.
            key (str): the path for the layer params.

        Returns:
            None

        """
        for i, ip in enumerate(self.params):
            df_params = pandas.DataFrame(be.to_numpy_array(ip))
            store.put(os.path.join(key, 'parameters', 'key'+str(i)), df_params)

    def load_params(self, store, key):
        """
        Load the parameters from an HDFStore.

        Notes:
            Performs an IO operation.

        Args:
            store (pandas.HDFStore): the readable stream for the params.
            key (str): the path for the layer params.

        Returns:
            None

        """
        params = []
        for i, ip in enumerate(self.params):
            params.append(be.float_tensor(
                store.get(os.path.join(key, 'parameters', 'key'+str(i))).as_matrix()
            ).squeeze()) # collapse trivial dimensions to a vector
        self.params = self.params.__class__(*params)

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
            Modifies the parameters of the layer in place.

        Args:
            None

        Returns:
            None

        """
        for param_name in self.constraints:
            self.constraints[param_name](getattr(self.params, param_name))

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
                   getattr(self.params, param_name)
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
                getattr(self.params, param_name))

    def parameter_step(self, deltas):
        """
        Update the values of the parameters:

        layer.params.name -= deltas.name

        Notes:
            Modifies the elements of the layer.params attribute in place.

        Args:
            deltas (dict): {param_name: tensor (update)}

        Returns:
            None

        """
        self.params = be.mapzip(be.subtract, deltas, self.params)
        self.enforce_constraints()


ParamsWeights = namedtuple("ParamsWeights", ["matrix"])

class Weights(Layer):
    """Layer class for weights"""

    def __init__(self, shape):
        """
        Create a weight layer.

        Notes:
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
        self.params = ParamsWeights(0.01 * be.randn(shape))

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

        A convenience method for accessing layer.params.matrix
        with a shorter syntax.

        Args:
            None

        Returns:
            tensor: weight matrix

        """
        return self.params.matrix

    def W_T(self):
        """
        Get the transpose of the weight matrix.

        A convenience method for accessing the transpose of
        layer.params.matrix with a shorter syntax.

        Args:
            None

        Returns:
            tensor: transpose of weight matrix

        """
        return be.transpose(self.params.matrix)

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
        derivs = ParamsWeights(
            self.get_penalty_grad(-be.batch_outer(vis, hid) / len(vis),
                                  "matrix"))
        return derivs
    
    def _grad_GFE(self, vis, hid):
        """
        Gradient of the Gibbs free energy associated with this layer

        Args:
            vis (magnetization object): magnetization of the lower layer linked to w
            hid (magnetization objet): magnetization of the upper layer linked to w
        """
        return ParamsWeights(-be.outer(vis.a(), hid.a()) - \
          be.multiply(self.params.matrix, be.outer(vis.c(), hid.c())))

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


ParamsGaussian = namedtuple("ParamsGaussian", ["loc", "log_var"])

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
        self.params = ParamsGaussian(be.zeros(self.len), be.zeros(self.len))

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
        diff = vis - be.broadcast(self.params.loc, vis)
        result = be.square(diff) / be.broadcast(scale, vis)
        return 0.5 * be.mean(result, axis=1)

    def log_partition_function(self, phi):
       """
       Compute the logarithm of the partition function of the layer
       with external field phi.

       Let u_i and s_i be the loc and scale parameters of unit i.
       Let phi_i be an external field

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
       logZ += be.multiply(scale, be.square(phi)) / 2
       logZ += be.log(be.broadcast(scale, phi)) / 2
       return logZ

    # TODO: What is this function? 
    # The docstring isn't consistent with what it does
    # and the variable names are not informative
    def augmented_log_partition_function(self, B, A):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi augmented with a quadratic factor.

        Let a_i be the loc parameter of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = Tr_{x_i} exp( a_i x_i + B_i x_i - A_i x_i^2)
        = 1 + \exp(a_i + B_i - A_i)

        log(Z_i) = softplus(a_i + B_i - A_i)

        Args:
            phi (tensor (num_samples, num_units)): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        return be.softplus(be.add(self.params.loc, be.subtract(A,B)))

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
        mean += be.broadcast(self.params.loc, mean)
        var = be.broadcast(be.exp(self.params.log_var), mean)
        if beta is not None:
            var = be.divide(beta, var)
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
        For a Gaussian layer, draws from a normal distribution
        with the mean and variance determined from the params attribute.

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            shape = be.shape(array_or_shape)
        except Exception:
            shape = array_or_shape

        mean = self.params.loc
        var = be.exp(self.params.log_var)
        r = self.rand(shape)

        return be.add(mean, be.multiply(be.sqrt(var), r))


ParamsIsing = namedtuple("ParamsIsing", ["loc"])

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
        self.params = ParamsIsing(be.zeros(self.len))

    def get_config(self):
        """
        Get the configuration dictionary of the Ising layer.

        Args:
            None:

        Returns:
            configuratiom (dict):

        """
        base_config = self.get_base_config()
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
        return -be.dot(data, self.params.loc)

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let a_i be the loc parameter of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = Tr_{x_i} exp( a_i x_i + phi_i x_i)
        = 2 cosh(a_i + phi_i)

        log(Z_i) = logcosh(a_i + phi_i)

        Args:
            phi (tensor (num_samples, num_units)): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        return be.logcosh(be.add(self.params.loc, phi))


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
        # get the current value of the first moment
        x = be.tanh(self.params.loc)

        # update the sample sizes
        n = len(data)
        new_sample_size = n + self.sample_size

        # updat the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the class attributes
        self.params = ParamsIsing(be.atanh(x))
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the parameters of the layer.
        Does nothing for the Ising layer.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        pass

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

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        loc = -be.mean(vis, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')
        return ParamsIsing(loc)

    def _conditional_params(self, scaled_units, weights, beta=None):
        """
        Compute the parameters of the layer conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor: conditional parameters

        """
        field = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            field += be.dot(scaled_units[i], weights[i])
        field += be.broadcast(self.params.loc, field)
        if beta is not None:
            field = be.multiply(beta, field)
        return field

    def conditional_mode(self, scaled_units, weights, beta=None):
        """
        Compute the mode of the distribution conditioned on the state
        of the connected layers.

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
        field = self._conditional_params(scaled_units, weights, beta)
        return 2 * be.float_tensor(field > 0) - 1

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
        field = self._conditional_params(scaled_units, weights, beta)
        return be.tanh(field)

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
        field = self._conditional_params(scaled_units, weights, beta)
        p = be.expit(field)
        r = self.rand(be.shape(p))
        return 2 * be.float_tensor(r < p) - 1

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For an Ising layer, draws -1 or +1 with the field determined
        by the params attribute.

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            shape = be.shape(array_or_shape)
        except Exception:
            shape = array_or_shape

        r = self.rand(shape)
        p = be.expit(be.broadcast(self.params.loc, r))
        return 2 * be.float_tensor(r < p) - 1


ParamsBernoulli = namedtuple("ParamsBernoulli", ["loc"])

#TODO: make interface that this must implement
class MagnetizationBernoulli(object):
    def __init__(self, a_0):
        self._a = a_0

    def __iter__(self):
        self.beginning = True
        return self

    def __next__(self):
        if self.beginning == False:
            raise StopIteration
        else:
            self.beginning = False
            return self._a

    def a(self):
        return self._a
    def c(self):
        return self._a - be.square(self._a)

    # TODO: should these really be here?
    def _grad_GFE_update_down(self, mag_lower, mag, w, ww):
        self._a -= be.dot(mag_lower.a(), w) + \
                be.multiply(be.dot(mag_lower.c(), ww),
                0.5 - mag.a())

    def _grad_GFE_update_up(self, mag, mag_upper, w, ww):
        self._a -= be.dot(w, mag_upper.a()) + \
                be.multiply(0.5 - mag.a(),
                be.dot(ww, mag_upper.c()))

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
        self.params = ParamsBernoulli(be.zeros(self.len))

    def get_zero_magnetization(self):
        return MagnetizationBernoulli(be.zeros(self.len))

    def get_random_magnetization(self):
        return MagnetizationBernoulli(be.rand((self.len,)))

    def get_config(self):
        """
        Get the configuration dictionary of the Bernoulli layer.

        Args:
            None:

        Returns:
            configuratiom (dict):

        """
        base_config = self.get_base_config()
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
        return -be.dot(data, self.params.loc)

    def log_partition_function(self, B, A):
        """
        Compute the logarithm of the partition function of the layer
        with external field B augmented with a quadratic, diagonal interaction A.

        Let a_i be the loc parameter of unit i.
        Let B_i be a local field
        Let A_i be a diagonal quadratic interaction

        Z_i = Tr_{x_i} exp( a_i x_i + B_i x_i - A_i x_i^2)
        = 1 + \exp(a_i + B_i - A_i)

        log(Z_i) = softplus(a_i + B_i - A_i)

        Args:
            A (tensor (num_samples, num_units)): external field
            B (tensor (num_samples, num_units)): diagonal quadratic external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        return be.softplus(be.add(self.params.loc, be.subtract(A,B)))

    def _grad_log_partition_function(self, B, A):
        """
        Compute the gradient of the logarithm of the partition function of the layer
        with external fields B,A as above.

        (d_a_i)softplus(a_i + B_i - A_i) = expit(a_i + B_i - A_i)

        Args:
            A (tensor (num_samples, num_units)): external field
            B (tensor (num_samples, num_units)): diagonal quadratic external field

        Returns:
            (d_a_i) logZ (tensor (num_samples, num_units)): gradient of the log partition function

        """
        return be.expit(be.add(be.unsqueeze(self.params.loc,0), be.subtract(A,B)))

    def grad_log_partition_function(self, B, A):
        """
        Compute the gradient of the logarithm of the partition function with respect to
        its local field parameter with external field B and quadratic interaction A.

        (d_a_i)softplus(a_i + B_i - A_i) = expit(a_i + B_i - A_i)

        Args:
            A (tensor (num_samples, num_units)): external field
            B (tensor (num_samples, num_units)): diagonal quadratic external field

        Returns:
            (d_a_i) logZ (tensor (num_samples, num_units)): gradient of the log partition function

        """
        return ParamsBernoulli(be.mean(self._grad_log_partition_function(B,A), axis=0))

    def _gibbs_lagrange_multipliers_1st_moment(self, mag):
        """
        The Lagrange multipliers associated with the first moment of the spins.

        Args:
            mag (magnetization object): magnetization of the layer
        """
        return be.subtract(self.params.loc, be.log(be.divide(1 - mag._a, mag._a)))

    def _gibbs_lagrange_multipliers_2nd_moment(self, mag):
        """
        The Lagrange multipliers associated with the second moment of the spins.
        For a Bernoulli layer this is strictly zero

        Args:
            mag (magnetization object): magnetization of the layer
        """
        return be.zeros_like(mag._a)

    def _gibbs_free_energy_entropy_term(self, B, A, mag):
        """
        The TAP-0 Gibbs free energy term associated strictly with this layer

        Args:
            B (float tensor like magnetization.a): 1st moment Lagrange multipler field
            A (float tensor like magnetization.a): strictly zero for Bernoulli layers
            mag (magnetization object): magnetization of the layer
        """
        return -be.tsum(self.log_partition_function(B, A)) + \
                be.dot(B, mag._a) + be.dot(A, mag._a)

    def _grad_magnetization_GFE(self, mag):
        """
        Gradient of the Gibbs free energy with respect to the magnetization
        associated strictly with this layer

        Args:
            mag (magnetization object): magnetization of the layer
        """
        return MagnetizationBernoulli(be.log(be.divide(1.0 - mag._a, mag._a)) - \
                                      self.params.loc)

    def _grad_loc_GFE(self, mag):
        """
        Gradient of the Gibbs free energy with respect to local field parameters

        Args:
            mag (magnetization object): magnetization of the layer
        """
        return ParamsBernoulli(-mag._a)

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
        # get the current value of the first moment
        x = be.expit(self.params.loc)

        # update the sample size
        n = len(data)
        new_sample_size = n + self.sample_size

        # update the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the class attributes
        self.params = ParamsBernoulli(be.logit(x))
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the parameters of the layer.
        Does nothing for the Bernoulli layer.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        pass

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

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        loc = -be.mean(vis, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')
        return ParamsBernoulli(loc)

    def _conditional_params(self, scaled_units, weights, beta=None):
        """
        Compute the parameters of the layer conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor: conditional parameters

        """
        field = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            field += be.dot(scaled_units[i], weights[i])
        field += be.broadcast(self.params.loc, field)
        if beta is not None:
            field = be.multiply(beta, field)
        return field

    def conditional_mode(self, scaled_units, weights, beta=None):
        """
        Compute the mode of the distribution conditioned on the state
        of the connected layers.

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
        field = self._conditional_params(scaled_units, weights, beta)
        return be.float_tensor(field > 0.0)

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
        field = self._conditional_params(scaled_units, weights, beta)
        return be.expit(field)

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
        field = self._conditional_params(scaled_units, weights, beta)
        p = be.expit(field)
        r = self.rand(be.shape(p))
        return be.float_tensor(r < p)

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For a Bernoulli layer, draws 0 or 1 with the field determined
        by the params attribute.

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            shape = be.shape(array_or_shape)
        except Exception:
            shape = array_or_shape

        r = self.rand(shape)
        p = be.expit(be.broadcast(self.params.loc, r))
        return be.float_tensor(r < p)


ParamsExponential = namedtuple("ParamsExponential", ["loc"])

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
        self.params = ParamsExponential(be.zeros(self.len))

    def get_config(self):
        """
        Get the configuration dictionary of the Exponential layer.

        Args:
            None:

        Returns:
            configuratiom (dict):

        """
        base_config = self.get_base_config()
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
        return be.dot(data, self.params.loc)

    def log_partition_function(self, phi):
        """
        Compute the logarithm of the partition function of the layer
        with external field phi.

        Let a_i be the loc parameter of unit i.
        Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.

        Z_i = Tr_{x_i} exp( -a_i x_i + phi_i x_i)
        = 1 / (a_i - phi_i)

        log(Z_i) = -log(a_i - phi_i)

        Args:
            phi (tensor (num_samples, num_units)): external field

        Returns:
            logZ (tensor, num_samples, num_units)): log partition function

        """
        return -be.log(be.subtract(self.params.loc, phi))

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
        # get the current value of the first moment
        x = be.reciprocal(self.params.loc)

        # update the sample size
        n = len(data)
        new_sample_size = n + self.sample_size

        # update the first moment
        x *= self.sample_size / new_sample_size
        x += n * be.mean(data, axis=0) / new_sample_size

        # update the class attributes
        self.params = ParamsExponential(be.reciprocal(x))
        self.sample_size = new_sample_size

    def shrink_parameters(self, shrinkage=1):
        """
        Apply shrinkage to the parameters of the layer.
        Does nothing for the Exponential layer.

        Args:
            shrinkage (float \in [0,1]): the amount of shrinkage to apply

        Returns:
            None

        """
        pass

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

    def derivatives(self, vis, hid, weights, beta=None):
        """
        Compute the derivatives of the layer parameters.

        Args:
            vis (tensor (num_samples, num_units)):
                The values of the visible units.
            hid list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the hidden units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        loc = be.mean(vis, axis=0)
        loc = self.get_penalty_grad(loc, 'loc')
        return ParamsExponential(loc)

    def _conditional_params(self, scaled_units, weights, beta=None):
        """
        Compute the parameters of the layer conditioned on the state
        of the connected layers.

        Args:
            scaled_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor: conditional parameters

        """
        rate = -be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            rate -= be.dot(scaled_units[i], weights[i])
        rate += be.broadcast(self.params.loc, rate)
        if beta is not None:
            rate = be.multiply(beta, rate)
        return rate

    def conditional_mode(self, scaled_units, weights, beta=None):
        """
        Compute the mode of the distribution conditioned on the state
        of the connected layers.

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
        raise NotImplementedError("Exponential distribution has no mode.")

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
        rate = self._conditional_params(scaled_units, weights, beta)
        return be.reciprocal(rate)

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
        rate = self._conditional_params(scaled_units, weights, beta)
        r = self.rand(be.shape(rate))
        return -be.log(r) / rate

    def random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For an Exponential layer, draws from the exponential distribution
        with the rate determined by the params attribute.

        Used for generating initial configurations for Monte Carlo runs.

        Args:
            array_or_shape (array or shape tuple):
                If tuple, then this is taken to be the shape.
                If array, then its shape is used.

        Returns:
            tensor: Random sample with desired shape.

        """
        try:
            shape = be.shape(array_or_shape)
        except Exception:
            shape = array_or_shape

        r = self.rand(shape)
        return be.divide(self.params.loc, -be.log(r))



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
