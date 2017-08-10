from collections import namedtuple

from .. import backends as be
from .. import math_utils
from .layer import Layer, CumulantsTAP

ParamsGaussian = namedtuple("ParamsGaussian", ["loc", "log_var"])

class GaussianLayer(Layer):
    """
    Layer with Gaussian units.

    """
    def __init__(self, num_units, dropout_p=0.0):
        """
        Create a layer with Gaussian units.

        Args:
            num_units (int): the size of the layer
            dropout_p (float): the probability that each unit is dropped out

        Returns:
            Gaussian layer

        """
        super().__init__(num_units, dropout_p)

        self.rand = be.randn
        self.params = ParamsGaussian(be.zeros(self.len), be.zeros(self.len))
        self.moments = math_utils.MeanVarianceArrayCalculator()

    #
    # Methods for the TAP approximation
    #

    def get_magnetization(self, mean):
        """
        Compute a CumulantsTAP object for the GaussianLayer.

        Args:
            expect (tensor (num_units,)): expected values of the units

        returns:
            CumulantsTAP

        """
        raise NotImplementedError

    def get_zero_magnetization(self):
        """
        Create a layer magnetization with zero expectations.

        Args:
            None

        Returns:
            CumulantsTAP

        """
        raise NotImplementedError

    def get_random_magnetization(self, epsilon=be.float_scalar(0.005)):
        """
        Create a layer magnetization with random expectations.

        Args:
            None

        Returns:
            CumulantsTAP

        """
        raise NotImplementedError

    def clip_magnetization(self, magnetization, a_min=be.float_scalar(1e-6),
                           a_max=be.float_scalar(1 - 1e-6)):
        """
        Clip the mean of the mean of a CumulantsTAP object.

        Args:
            magnetization (CumulantsTAP) to clip
            a_min (float): the minimum value
            a_max (float): the maximum value

        Returns:
            clipped magnetization (CumulantsTAP)

        """
        raise NotImplementedError

    def log_partition_function(self, external_field, quadratic_field):
        """
        Compute the logarithm of the partition function of the layer
        with external field (B) and quadratic field (A).

        Args:
            external_field (tensor (num_samples, num_units)): external field
            quadratic_field (tensor (num_samples, num_units)): quadratic field

        Returns:
            logZ (tensor (num_samples, num_units)): log partition function

        """
        raise NotImplementedError

    def grad_log_partition_function(self, external_field, quadratic_field):
        """
        Compute the gradient of the logarithm of the partition function with respect to
        its local field parameter with external field (B) and quadratic field (A).

        Note: This function returns the mean parameters over a minibatch of input fields

        Args:
            external_field (tensor (num_samples, num_units)): external field
            quadratic_field (tensor (num_samples, num_units)): quadratic field

        Returns:
            (d_a_i) logZ (tensor (num_samples, num_units)): gradient of the log partition function

        """
        raise NotImplementedError

    def lagrange_multiplers(self, cumulants):
        """
        The Lagrange multipliers associated with the first and second
        cumulants of the units.

        Args:
            cumulants (CumulantsTAP object): cumulants

        Returns:
            lagrange multipliers (CumulantsTAP)

        """
        raise NotImplementedError

    def TAP_entropy(self, lagrange, cumulants):
        """
        The TAP-0 Gibbs free energy term associated strictly with this layer

        Args:
            lagrange (CumulantsTAP): Lagrange multiplers
            cumulants (CumulantsTAP): magnetization of the layer

        Returns:
            (float): 0th order term of Gibbs free energy
        """
        raise NotImplementedError

    def TAP_magnetization_grad(self, mag, connected_mag, connected_weights):
        """
        Gradient of the Gibbs free energy with respect to the magnetization
        associated strictly with this layer.

        Args:
            mag (CumulantsTAP object): magnetization of the layer
            connected_mag list[CumulantsTAP]: magnetizations of the connected layers
            connected weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.

        Return:
            gradient of GFE w.r.t. magnetization (CumulantsTAP)

        """
        raise NotImplementedError

    def GFE_derivatives(self, cumulants, penalize=True):
        """
        Gradient of the Gibbs free energy with respect to local field parameters

        Args:
            cumulants (CumulantsTAP object): magnetization of the layer

        Returns:
            gradient parameters (ParamsGaussian): gradient w.r.t. local fields of GFE
        """
        raise NotImplementedError

    #
    # Methods for sampling and sample-based training
    #

    def energy(self, units):
        """
        Compute the energy of the Gaussian layer.

        For sample k,
        E_k = \frac{1}{2} \sum_i \frac{(v_i - loc_i)**2}{var_i}

        Args:
            units (tensor (num_samples, num_units)): values of units

        Returns:
            tensor (num_samples,): energy per sample

        """
        scale = be.exp(self.params.log_var)
        diff = units - self.params.loc
        result = be.square(diff) / scale
        return 0.5 * be.mean(result, axis=1)

    def log_partition_function(self, external_field):
       """
       Compute the logarithm of the partition function of the layer
       with external field (phi).

       Let u_i and s_i be the loc and scale parameters of unit i.
       Let phi_i be an external field

       Z_i = \int d x_i exp( -(x_i - u_i)^2 / (2 s_i^2) + \phi_i x_i)
       = exp(b_i u_i + b_i^2 s_i^2 / 2) sqrt(2 pi) s_i

       log(Z_i) = log(s_i) + phi_i u_i + phi_i^2 s_i^2 / 2

       Args:
           external_field (tensor (num_samples, num_units)z0: external field

       Returns:
           logZ (tensor, (num_samples, num_units)): log partition function

       """
       variance = be.exp(self.params.log_var)
       logZ = be.multiply(self.params.loc, external_field)
       logZ += be.multiply(variance, be.square(external_field)) / 2
       logZ += be.log(variance) / 2
       return logZ

    def online_param_update(self, units):
        """
        Update the parameters using an observed batch of data.
        Used for initializing the layer parameters.

        Notes:
            Modifies layer.params in place.

        Args:
            units (tensor (num_samples, num_units)): observed values for units

        Returns:
            None

        """
        self.moments.update(units)
        self.set_params(ParamsGaussian(self.moments.mean, be.log(self.moments.var)))

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
        self.set_params(ParamsGaussian(self.params.loc, be.log(var)))

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

    def derivatives(self, units, connected_units, connected_weights, penalize=True):
        """
        Compute the derivatives of the layer parameters.

        Args:
            units (tensor (num_samples, num_units)):
                The values of the layer units.
            connected_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            connected_weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        # initialize tensors for the location and scale derivatives
        loc = be.zeros(self.len),
        log_var = be.zeros(self.len)

        # compute the derivative with respect to the location parameter
        v_scaled = self.rescale(units)
        loc = -be.mean(v_scaled, axis=0)

        if penalize:
            loc = self.get_penalty_grad(loc, 'loc')

        # compute the derivative with respect to the scale parameter
        log_var = -0.5 * be.mean(be.square(be.subtract(
            self.params.loc, units)), axis=0)
        for i in range(len(connected_units)):
            log_var += be.batch_dot(connected_units[i], connected_weights[i], units, axis=0) / len(units)
        log_var = self.rescale(log_var)

        if penalize:
            log_var = self.get_penalty_grad(log_var, 'log_var')

        # return the derivatives in a namedtuple
        return [ParamsGaussian(loc, log_var)]

    def zero_derivatives(self):
        """
        Return an object like the derivatives that is filled with zeros.

        Args:
            None

        Returns:
            derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)

        """
        return [be.apply(be.zeros_like, self.params)]

    def random_derivatives(self):
        """
        Return an object like the derivatives that is filled with random floats.

        Args:
            None

        Returns:
            derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)

        """
        return [be.apply(be.rand_like, self.params)]

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
        mean += self.params.loc
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

    def onehot(self, n):
        """
        Generate an (n x n) tensor where each row has one unit with maximum
        activation and all other units with minimum activation.

        Args:
            n (int): the number of units

        Returns:
            tensor (n, n)

        """
        std = be.sqrt(be.exp(self.params.log_var))
        return be.diagonal_matrix(3*std)
