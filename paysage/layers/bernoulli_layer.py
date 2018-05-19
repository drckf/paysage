from collections import namedtuple

from .. import backends as be
from .layer import Layer, CumulantsTAP

ParamsBernoulli = namedtuple("ParamsBernoulli", ["loc"])

class BernoulliLayer(Layer):
    """
    Layer with Bernoulli units (i.e., 0 or +1).

    """
    def __init__(self, num_units, center=False):
        """
        Create a layer with Bernoulli units.

        Args:
            num_units (int): the size of the layer
            center (bool): whether to center the layer

        Returns:
            Bernoulli layer

        """
        super().__init__(num_units, center)

        self.rand = be.rand
        self.params = ParamsBernoulli(be.zeros(self.len))

    #
    # Methods for the TAP approximation
    #

    def get_magnetization(self, mean):
        """
        Compute a CumulantsTAP object for the BernoulliLayer.

        Args:
            mean (tensor (num_units,)): expected values of the units

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
        return self.get_magnetization(be.zeros_like(self.params[0]))

    def get_random_magnetization(self, num_samples=1, epsilon=be.float_scalar(1e-6)):
        """
        Create a layer magnetization with random expectations.

        Args:
            num_samples (int>0): number of random samples to draw
            epsilon (float): bound away from [0,1] in which to draw magnetization values

        Returns:
            CumulantsTAP

        """
        # If num_samples == 1 we do not vectorize computations over a sampling set
        # for the sake of performance
        if num_samples > 1:
            return self.get_magnetization(be.clip(be.rand((num_samples,self.len,)),
                a_min=epsilon, a_max=be.float_scalar(1-epsilon)))
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

    def clip_magnetization_(self, magnetization, a_min=be.float_scalar(1e-6),
                           a_max=be.float_scalar(1 - 1e-6)):
        """
        Clip the mean of the mean of a CumulantsTAP object.

        Args:
            magnetization (CumulantsTAP) to clip
            a_min (float): the minimum value
            a_max (float): the maximum value

        Returns:
            None

        """
        be.clip_(magnetization.mean[:],  a_min=a_min, a_max=a_max)
        magnetization.variance[:] = magnetization.mean - be.square(magnetization.mean)

    def log_partition_function(self, external_field, quadratic_field):
        """
        Compute the logarithm of the partition function of the layer
        with external field (B) and quadratic field (A).

        Let a_i be the loc parameter of unit i.
        Let B_i be an external field
        Let A_i be a quadratic field

        Z_i = Tr_{x_i} exp( a_i x_i + B_i x_i + A_i x_i^2)
        = 1 + \exp(a_i + B_i + A_i)

        log(Z_i) = softplus(a_i + B_i + A_i)

        Args:
            external_field (tensor (num_samples, num_units)): external field
            quadratic_field (tensor (num_samples, num_units)): quadratic field

        Returns:
            logZ (tensor (num_samples, num_units)): log partition function

        """
        return be.softplus(self.params.loc + quadratic_field + external_field)

    def lagrange_multipliers_analytic(self, cumulants):
        """
        Return the Lagrange multipliers (at beta=0) according to the starionarity
            conditions {d/da(GibbsFE)=0, d/dc(GibbsFE)=0} at beta=0.

        Args:
            cumulants (CumulantsTAP object): layer magnetization cumulants

        Returns:
            lagrange multipliers (CumulantsTAP)

        """
        mean = be.subtract(self.params.loc, be.logit(cumulants.mean))
        variance = be.zeros_like(cumulants.variance)
        return CumulantsTAP(mean, variance)

    def update_lagrange_multipliers_(self, cumulants, lagrange_multipliers,
                                     connected_cumulants,
                                     rescaled_connected_weights,
                                     rescaled_connected_weights_sq):
        """
        Update, in-place, the Lagrange multipliers with respect to the TAP2 approximation
        of the GFE as in

        Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,
        and Florent Krzakala
        "A Deterministic and Generalized Framework for Unsupervised Learning
        with Restricted Boltzmann Machines"

        Args:
            cumulants (CumulantsTAP): layer magnetization cumulants
            lagrange_multipliers (CumulantsTAP)
            connected_cumulants (CumulantsTAP): connected magnetization cumulants
            rescaled_connected_weights (list[tensor, (num_connected_units, num_units)]):
                The weights connecting the layers.
            rescaled_connected_weights_sq (list[tensor, (num_connected_units, num_units)]):
                The cached squares of weights connecting the layers.
                (unused on Bernoulli layer)

        Returns:
            None

        """
        lagrange_multipliers.variance[:] = be.zeros_like(lagrange_multipliers.variance)
        lagrange_multipliers.mean[:] = be.zeros_like(lagrange_multipliers.mean)
        for l in range(len(connected_cumulants)):
            # let len(mean) = N and len(connected_mag[l].mean) = N_l
            # weights[l] is a matrix of shape (N_l, N)
            w_l = rescaled_connected_weights[l]
            w2_l = rescaled_connected_weights_sq[l]
            lagrange_multipliers.mean[:] += \
                be.dot(connected_cumulants[l].mean, w_l) + \
                be.multiply(be.dot(connected_cumulants[l].variance, w2_l),
                                   0.5 - cumulants.mean)

    def TAP_entropy(self, cumulants):
        """
        The TAP-0 Gibbs free energy term associated strictly with this layer

        Args:
            cumulants (CumulantsTAP): magnetization of the layer

        Returns:
            (float): 0th order term of Gibbs free energy
        """
        # this quadratic approximation is 2x faster:
        #a = be.float_scalar(1.06*2.77258872224)
        #u = be.float_scalar(1.06*-0.69314718056)
        #return be.tsum(be.add(u, a * be.square(be.subtract(0.5, cumulants.mean)))) - \
        #       be.dot(self.params.loc, cumulants.mean)
        alias = 1.0-cumulants.mean
        return be.dot(cumulants.mean, be.log(cumulants.mean)) + \
               be.dot(alias, be.log(alias)) - \
               be.dot(self.params.loc, cumulants.mean)

    def TAP_magnetization_grad(self, cumulants,
                               connected_cumulants, rescaled_connected_weights,
                               rescaled_connected_weights_sq):
        """
        Gradient of the Gibbs free energy with respect to the magnetization
        associated strictly with this layer.

        Args:
            cumulants (CumulantsTAP): magnetization of the layer
            connected_cumulants (list[CumulantsTAP]): magnetizations of the connected layers
            rescaled_connected_weights (list[tensor, (num_connected_units, num_units)]):
                The weights connecting the layers.
            rescaled_connected_weights_sq (list[tensor, (num_connected_units, num_units)]):
                The cached squares of weights connecting the layers.

        Return:
            gradient of GFE w.r.t. magnetization (CumulantsTAP)

        """
        mean = be.logit(cumulants.mean) - self.params.loc
        variance = be.zeros_like(mean)

        for l in range(len(connected_cumulants)):
            # let len(mean) = N and len(connected_cumulants[l].mean) = N_l
            # weights[l] is a matrix of shape (N_l, N)
            w_l = rescaled_connected_weights[l]
            w2_l = rescaled_connected_weights_sq[l]

            mean -= be.dot(connected_cumulants[l].mean, w_l) + \
                    be.multiply(be.dot(connected_cumulants[l].variance, w2_l),
                                0.5 - cumulants.mean)

        return CumulantsTAP(mean, variance)

    def self_consistent_update_(self, cumulants, lagrange_multipliers):
        """
        Applies self-consistent TAP update to the layer's magnetization. This formula
         is analytically computed --not based on a 2-term truncation of the Gibbs FE.

        Args:
            cumulants (CumulantsTAP object): magnetization of the layer
            lagrange_multipliers (CumulantsTAP object)

        Returns:
            None
        """
        cumulants.mean[:] = be.expit(self.params.loc + lagrange_multipliers.mean)
        cumulants.variance[:] = cumulants.mean - be.square(cumulants.mean)

    def GFE_derivatives(self, cumulants, connected_cumulants=None,
                        rescaled_connected_weights=None,
                        rescaled_connected_weights_sq=None):
        """
        Gradient of the Gibbs free energy with respect to local field parameters

        Args:
            cumulants (CumulantsTAP object): magnetization of the layer

        Returns:
            gradient parameters (ParamsBernoulli): gradient w.r.t. local fields of GFE
        """
        return [ParamsBernoulli(-cumulants.mean)]

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
        self.set_params([ParamsBernoulli(be.logit(self.moments.mean))])

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
        if not self.center:
            return observations
        return be.subtract(self.get_center(), observations)

    def rescale_cumulants(self, cumulants):
        """
        Rescales the cumulants associated with the layer.
         Trivial for the Bernoulli layer.

        Args:
            cumulants (CumulantsTAP)

        Returns:
            rescaled cumulants (CumulantsTAP)
        """
        return cumulants

    def reciprocal_scale(self):
        """
        Returns a tensor of shape (num_units) providing a reciprocal scale for each unit

        Args:
            None
        Returns:
            reciproical scale (tensor)
        """
        return be.ones_like(self.params[0])

    def derivatives(self, units, connected_units, connected_weights,
                    penalize=True, weighting_function=be.do_nothing):
        """
        Compute the derivatives of the layer parameters.

        Args:
            units (tensor (num_samples, num_units)):
                The values of the layer units.
            connected_units list[tensor (num_samples, num_connected_units)]:
                The rescaled values of the connected units.
            connected_weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            penalize (bool): whether to add a penalty term.
            weighting_function (function): a weighting function to apply
                to units when computing the gradient.

        Returns:
            grad (namedtuple): param_name: tensor (contains gradient)

        """
        loc = -be.mean(weighting_function(units), axis=0)
        if penalize:
            loc = self.get_penalty_grad(loc, 'loc')
        return [ParamsBernoulli(loc)]

    def zero_derivatives(self):
        """
        Return an object like the derivatives that is filled with zeros.

        Args:
            None

        Returns:
            derivs (List[namedtuple]): List[param_name: tensor] (contains gradient)

        """
        return [be.apply(be.zeros_like, self.params)]

    def random_derivatives(self):
        """
        Return an object like the derivatives that is filled with random floats.

        Args:
            None

        Returns:
            derivs (List[namedtuple]): List[param_name: tensor] (contains gradient)

        """
        return [be.apply(be.rand_like, self.params)]

    def conditional_params(self, scaled_units, weights, beta=None):
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
        field = self.conditional_params(scaled_units, weights, beta)
        return be.cast_float(field > 0.0)

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
        field = self.conditional_params(scaled_units, weights, beta)
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
        field = self.conditional_params(scaled_units, weights, beta)
        p = be.expit(field)
        r = self.rand(be.shape(p))
        return be.cast_float(r < p)

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
        return be.cast_float(r < p)

    def envelope_random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For a Bernoulli layer, draws 0 or 1 from a bernoulli layer with mean
        self.moments.mean

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
        p = be.expit(self.moments.mean)
        return be.cast_float(r < p)
