from collections import namedtuple

from .. import backends as be
from .. import math_utils
from .layer import Layer, CumulantsTAP

ParamsBernoulli = namedtuple("ParamsBernoulli", ["loc"])

class BernoulliLayer(Layer):
    """
    Layer with Bernoulli units (i.e., 0 or +1).

    """
    def __init__(self, num_units, dropout_p=0.0):
        """
        Create a layer with Bernoulli units.

        Args:
            num_units (int): the size of the layer
            dropout_p (float): the probability that each unit is dropped out

        Returns:
            Bernoulli layer

        """
        super().__init__(num_units, dropout_p)

        self.rand = be.rand
        self.params = ParamsBernoulli(be.zeros(self.len))
        self.moments = math_utils.MeanArrayCalculator()

    #
    # Methods for the TAP approximation
    #

    def get_magnetization(self, mean):
        """
        Compute a CumulantsTAP object for the BernoulliLayer.

        Args:
            expect (tensor (num_units,)): expected values of the units

        returns:
            CumulantsTAP

        """
        return CumulantsTAP(mean, mean - be.square(mean))

    def get_zero_magnetization(self):
        """
        Create a layer magnetization with zero expectations.

        Args:
            None

        Returns:
            CumulantsTAP

        """
        return self.get_magnetization(be.zeros(self.len))

    def get_random_magnetization(self, epsilon=be.float_scalar(0.005)):
        """
        Create a layer magnetization with random expectations.

        Args:
            None

        Returns:
            CumulantsTAP

        """
        return self.get_magnetization(be.clip(be.rand((self.len,)),
                a_min=epsilon, a_max=be.float_scalar(1-epsilon)))

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
        tmp = be.clip(magnetization.mean,  a_min=a_min, a_max=a_max)
        return self.get_magnetization(tmp)

    def log_partition_function(self, external_field, quadratic_field):
        """
        Compute the logarithm of the partition function of the layer
        with external field (B) and quadratic field (A).

        Let a_i be the loc parameter of unit i.
        Let B_i be an external field
        Let A_i be a quadratic field

        Z_i = Tr_{x_i} exp( a_i x_i + B_i x_i - A_i x_i^2)
        = 1 + \exp(a_i + B_i - A_i)

        log(Z_i) = softplus(a_i + B_i - A_i)

        Args:
            external_field (tensor (num_samples, num_units)): external field
            quadratic_field (tensor (num_samples, num_units)): quadratic field

        Returns:
            logZ (tensor (num_samples, num_units)): log partition function

        """
        return be.softplus(be.add(self.params.loc, be.subtract(quadratic_field, external_field)))

    def grad_log_partition_function(self, external_field, quadratic_field):
        """
        Compute the gradient of the logarithm of the partition function with respect to
        its local field parameter with external field (B) and quadratic field (A).

        (d_a_i)softplus(a_i + B_i - A_i) = expit(a_i + B_i - A_i)

        Note: This function returns the mean parameters over a minibatch of input fields

        Args:
            external_field (tensor (num_samples, num_units)): external field
            quadratic_field (tensor (num_samples, num_units)): quadratic field

        Returns:
            (d_a_i) logZ (tensor (num_samples, num_units)): gradient of the log partition function

        """
        tmp = be.expit(be.add(be.unsqueeze(self.params.loc,0), be.subtract(quadratic_field, external_field)))
        return ParamsBernoulli(be.mean(tmp, axis=0))

    def lagrange_multiplers(self, cumulants):
        """
        The Lagrange multipliers associated with the first and second
        cumulants of the units.

        Args:
            cumulants (CumulantsTAP object): cumulants

        Returns:
            lagrange multipliers (CumulantsTAP)

        """
        mean = be.subtract(self.params.loc, be.logit(cumulants.mean))
        variance = be.zeros_like(cumulants.variance)
        return CumulantsTAP(mean, variance)

    def TAP_entropy(self, lagrange, cumulants):
        """
        The TAP-0 Gibbs free energy term associated strictly with this layer

        Args:
            lagrange (CumulantsTAP): Lagrange multiplers
            cumulants (CumulantsTAP): magnetization of the layer

        Returns:
            (float): 0th order term of Gibbs free energy
        """
        return -be.tsum(self.log_partition_function(lagrange.mean, lagrange.variance)) + \
                be.dot(lagrange.mean, cumulants.mean) + be.dot(lagrange.variance, cumulants.mean)

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
        mean = be.logit(mag.mean) - self.params.loc
        variance = be.zeros_like(mean)

        for l in range(len(connected_mag)):
            # let len(mean) = N and len(connected_mag[l].mean) = N_l
            # weights[l] is a matrix of shape (N_l, N)
            w_l = connected_weights[l]
            w2_l = be.square(w_l)

            mean -= be.dot(connected_mag[l].mean, w_l) + \
                    be.multiply(be.dot(connected_mag[l].variance, w2_l), 0.5 - mag.mean)

        return CumulantsTAP(mean, variance)

    def GFE_derivatives(self, cumulants, penalize=True):
        """
        Gradient of the Gibbs free energy with respect to local field parameters

        Args:
            cumulants (CumulantsTAP object): magnetization of the layer

        Returns:
            gradient parameters (ParamsBernoulli): gradient w.r.t. local fields of GFE
        """
        tmp = -cumulants.mean
        if penalize:
            tmp = self.get_penalty_grad(tmp, "loc")
        return [ParamsBernoulli(tmp)]

    #
    # Methods for sampling and sample-based training
    #

    def energy(self, units):
        """
        Compute the energy of the Bernoulli layer.

        For sample k,
        E_k = -\sum_i loc_i * v_i

        Args:
            units (tensor (num_samples, num_units)): values of units

        Returns:
            tensor (num_samples,): energy per sample

        """
        return -be.dot(units, self.params.loc)

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
        self.moments.update(units, axis=0)
        self.set_params(ParamsBernoulli(be.logit(self.moments.mean)))

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
        Rescale is trivial for the Bernoulli layer.

        Args:
            observations (tensor (num_samples, num_units)):
                Values of the observed units.
        Returns:
            tensor: observations

        """
        return observations

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
        tmp = -be.mean(units, axis=0)
        if penalize:
            tmp = self.get_penalty_grad(tmp, 'loc')
        return [ParamsBernoulli(tmp)]

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
            weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            beta (tensor (num_samples, 1), optional):
                Inverse temperatures.

        Returns:
            tensor: conditional parameters

        """
        assert(len(scaled_units) == len(weights))
        field = be.dot(scaled_units[0], weights[0])
        for i in range(1, len(weights)):
            field += be.dot(scaled_units[i], weights[i])
        field += self.params.loc
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
        p = be.expit(self.params.loc)
        return be.float_tensor(r < p)

    def onehot(self, n):
        """
        Generate an (n x n) tensor where each row has one unit with maximum
        activation and all other units with minimum activation.

        Args:
            n (int): the number of units

        Returns:
            tensor (n, n)

        """
        return be.identity(n)
