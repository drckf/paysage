from collections import namedtuple
import math

from .. import backends as be
from .layer import Layer, CumulantsTAP

ParamsGaussian = namedtuple("ParamsGaussian", ["loc", "log_var"])

class GaussianLayer(Layer):
    """
    Layer with Gaussian units.

    """
    def __init__(self, num_units, center=False):
        """
        Create a layer with Gaussian units.

        Args:
            num_units (int): the size of the layer

        Returns:
            Gaussian layer

        """
        super().__init__(num_units, center)

        self.rand = be.randn
        self.params = ParamsGaussian(be.zeros(self.len), be.zeros(self.len))

    #
    # Methods for the TAP approximation
    #

    def get_zero_magnetization(self):
        """
        Create a layer magnetization with zero expectations.

        Args:
            None

        Returns:
            CumulantsTAP
        """
        return CumulantsTAP(be.zeros_like(self.params[0]), be.zeros_like(self.params[0]))

    def get_random_magnetization(self, num_samples=1):
        """
        Create a layer magnetization with random expectations.

        Args:
            None

        Returns:
            CumulantsTAP
        """
        if num_samples == 1:
            return CumulantsTAP(self.rand((self.len,)),
                                be.exp(self.rand((self.len,))))
        return CumulantsTAP(self.rand((num_samples, self.len,)),
                            be.exp(self.rand((num_samples, self.len,))))

    def clip_magnetization(self, magnetization):
        """
        Clip the variance at zero.

        Args:
            magnetization (CumulantsTAP) to clip

        Returns:
            clipped magnetization (CumulantsTAP)
        """
        return CumulantsTAP(magnetization.mean,
                            be.clip(magnetization.variance,  a_min=1e-6))

    def clip_magnetization_(self, magnetization):
        """
        Clip the variance at zero in place.

        Args:
            magnetization (CumulantsTAP) to clip

        Returns:
            None
        """
        be.clip_(magnetization.variance, a_min=1e-6)

    def log_partition_function(self, B, A):
        """
        Compute the logarithm of the partition function of the layer
        with external field B augmented with a diagonal quadratic interaction A.
        Let u_i and s_i be the loc and scale parameters of unit i.

        Z_i = \int d x_i exp( -(x_i - u_i)^2 / (2 s_i^2) + B_i x_i + A_i x_i^2)
            = exp((B_i u_i + A_i + B_i^2 s_i^2 / 2)/(1 - 2 s_i^2 A_i)) sqrt(2 pi) s_i
        log(Z_i) = (B_i u_i + A_i + B_i^2 s_i^2 / 2)/(1 - 2 s_i^2 A_i) *
                   1/2 log((2 pi s_i^2)/ (1 - 2 s_i^2 A_i))

        Args:
            A (tensor (num_samples, num_units)): external field
            B (tensor (num_samples, num_units)): diagonal quadratic external field

        Returns:
            logZ (tensor (num_samples, num_units)): log partition function
        """
        scale = be.exp(self.params.log_var)
        denom = 1.0 - 2.0 * be.multiply(scale, A)
        logZ = 0.5*be.log(be.divide(denom, be.broadcast(2.0*math.pi*scale, denom))) + \
               be.divide(denom, be.multiply(self.params.loc, B) + \
               be.multiply(be.square(self.params.loc), A) + \
               0.5*be.multiply(scale, be.square(B)))
        return logZ

    def lagrange_multipliers_approx(self, cumulants, connected_cumulants,
                                     rescaled_connected_weights,
                                     rescaled_connected_weights_sq):
        """
        Return the Lagrange multipliers according to a TAP2
        approximation as in

        Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,
        and Florent Krzakala
        "A Deterministic and Generalized Framework for Unsupervised Learning
        with Restricted Boltzmann Machines"

        Args:
            cumulants (CumulantsTAP object): layer magnetization cumulants
            connected_cumulants (CumulantsTAP object): connected magnetization cumulants
            rescaled_connected_weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            rescaled_connected_weights_sq list[tensor, (num_connected_units, num_units)]:
                The cached squares of weights connecting the layers.
                (unused on Bernoulli layer)

        Returns:
            lagrange multipliers (CumulantsTAP)
        """
        variance = be.zeros_like(cumulants.variance)
        for l in range(len(connected_cumulants)):
            # let len(variance) = N and len(connected_mag[l].variance) = N_l
            # w2 is a matrix of shape (N_l, N)
            w2 = rescaled_connected_weights_sq[l]
            variance += 0.5*be.dot(connected_cumulants[l].variance, w2)

        mean = be.zeros_like(cumulants.mean)
        for l in range(len(connected_cumulants)):
            # let len(mean) = N and len(connected_mag[l].mean) = N_l
            # weights[l] is a matrix of shape (N_l, N)
            w_l = rescaled_connected_weights[l]
            mean += -2.0*be.multiply(cumulants.mean, variance) + \
                    be.dot(connected_cumulants[l].mean, w_l)

        return CumulantsTAP(mean, variance)

    def lagrange_multipliers_analytic(self, cumulants):
        """
        Return the Lagrange multipliers (at beta=0) according to the starionarity
            conditions {d/da(GibbsFE)=0, d/dc(GibbsFE)=0} at beta=0.

        Args:
            cumulants (CumulantsTAP object): layer magnetization cumulants

        Returns:
            lagrange multipliers (CumulantsTAP)
        """
        scale = be.exp(self.params.log_var)
        clipd = be.clip(be.multiply(cumulants.variance, scale), a_min=1e-6)
        variance = \
            0.5 * be.divide(clipd, be.subtract(scale, cumulants.variance))
        mean = be.divide(clipd,
            be.multiply(scale, cumulants.mean) - \
            be.multiply(cumulants.variance, self.params.loc))

        return CumulantsTAP(mean, variance)

    def update_lagrange_multipliers_(self, cumulants, lagrange_multipliers,
                                     connected_cumulants, rescaled_connected_weights,
                                     rescaled_connected_weights_sq):
        """
        Update, in-place, the Lagrange multipliers with respect to the TAP2 approximation
        of the GFE as in

        Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,
        and Florent Krzakala
        "A Deterministic and Generalized Framework for Unsupervised Learning
        with Restricted Boltzmann Machines"

        Args:
            cumulants (CumulantsTAP object): layer magnetization cumulants
            lagrange_multipliers (CumulantsTAP object)
            connected_cumulants (CumulantsTAP object): connected magnetization cumulants
            rescaled_connected_weights list[tensor, (num_connected_units, num_units)]:
                The weights connecting the layers.
            rescaled_connected_weights_sq list[tensor, (num_connected_units, num_units)]:
                The cached squares of weights connecting the layers.
                (unused on Bernoulli layer)

        Returns:
            None
        """
        lagrange_multipliers.variance[:] = be.zeros_like(lagrange_multipliers.variance)
        for l in range(len(connected_cumulants)):
            # let len(variance) = N and len(connected_mag[l].variance) = N_l
            # w2 is a matrix of shape (N_l, N)
            w2 = rescaled_connected_weights_sq[l]
            lagrange_multipliers.variance[:] += 0.5*be.dot(connected_cumulants[l].variance, w2)

        lagrange_multipliers.mean[:] = be.zeros_like(lagrange_multipliers.mean)
        for l in range(len(connected_cumulants)):
            # let len(mean) = N and len(connected_mag[l].mean) = N_l
            # weights[l] is a matrix of shape (N_l, N)
            w_l = rescaled_connected_weights[l]
            lagrange_multipliers.mean[:] += \
                -2.0*be.multiply(cumulants.mean, lagrange_multipliers.variance) + \
                be.dot(connected_cumulants[l].mean, w_l)

    def TAP_entropy(self, cumulants):
        """
        The TAP-0 Gibbs free energy term associated strictly with this layer

        Args:
            cumulants (CumulantsTAP): magnetization of the layer

        Returns:
            (float): 0th order term of Gibbs free energy
        """
        scale = be.exp(self.params.log_var)
        return  be.tsum(-0.5*be.log(2.0*math.pi*cumulants.variance) + \
                be.divide(2.0*scale, be.square(be.subtract(cumulants.mean, self.params.loc)) + \
                          cumulants.variance - scale))

    def TAP_magnetization_grad(self, cumulants,
                               connected_cumulants, rescaled_connected_weights,
                               rescaled_connected_weights_sq):
        """
        Gradient of the Gibbs free energy with respect to the magnetization
        associated strictly with this layer.

        Args:
            cumulants (CumulantsTAP): magnetization of the layer
            connected_cumulants (list[CumulantsTAP]): magnetizations of the connected layers
            rescaled_connected weights (list[tensor, (num_connected_units, num_units)]):
                The weights connecting the layers.
            rescaled_connected_weights_sq (list[tensor, (num_connected_units, num_units)]):
                The cached squares of weights connecting the layers.

        Returns:
            gradient of GFE w.r.t. magnetization (CumulantsTAP)
        """
        scale = be.exp(self.params.log_var)
        mean = be.divide(scale, cumulants.mean - self.params.loc)
        clipd = be.clip(scale * cumulants.variance, a_min=1e-6)
        variance = be.divide(2.0* clipd, cumulants.variance - scale)

        for l in range(len(connected_cumulants)):
            # let len(mean) = N and len(connected_mag[l].mean) = N_l
            # weights[l] is a matrix of shape (N_l, N)
            w_l = rescaled_connected_weights[l]
            w2_l = rescaled_connected_weights_sq[l]

            mean -= be.dot(connected_cumulants[l].mean, w_l)
            variance -= 0.5*be.dot(connected_cumulants[l].variance, w2_l)

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
        scale = be.exp(self.params.log_var)
        denom = 1.0 - 2.0 * be.multiply(scale, lagrange_multipliers.variance)
        cumulants.mean[:] = be.divide(denom,
                      be.multiply(scale, lagrange_multipliers.mean) + self.params.loc)
        cumulants.variance[:] = be.divide(denom, scale)

    def GFE_derivatives(self, cumulants, connected_cumulants,
                        rescaled_connected_weights, rescaled_connected_weights_sq):
        """
        Gradient of the 0th order Gibbs free energy with respect to local field parameters.

        Args:
            cumulants (CumulantsTAP): magnetization of the layer
            connected_cumulants (list[CumulantsTAP]): magnetizations of the connected layers
            rescaled_connected_weights (list[tensor, (num_connected_units, num_units)]):
                The weights connecting the layers.
            rescaled_connected_weights_sq (list[tensor, (num_connected_units, num_units)]):
                The cached squares of weights connecting the layers.

        Returns:
            gradient parameters (ParamsGaussian): gradient w.r.t. local fields of GFE
        """
        scale = be.exp(self.params.log_var)

        # using analytic formula for lagrange multipliers as function of the cumulants:
        d_ulogZ = be.divide(scale, self.params.loc - cumulants.mean)

        d_lnslogZ = -be.divide(2.0*scale,
                               cumulants.variance + be.square(self.params.loc - cumulants.mean))

        for l in range(len(connected_cumulants)):
            # let len(mean) = N and len(connected_mag[l].mean) = N_l
            # weights[l] is a matrix of shape (N_l, N)
            w_l = rescaled_connected_weights[l]
            w2_l = rescaled_connected_weights_sq[l]

            d_lnslogZ += be.multiply(be.dot(connected_cumulants[l].mean, w_l),
                                     cumulants.mean)
            d_lnslogZ += be.multiply(be.dot(connected_cumulants[l].variance, w2_l),
                                     cumulants.variance)

        # Formula generated by using TAP2 approximation of lagrange multipliers
        #d_lnslogZ = be.divide(2.0*scale, cumulants.variance + be.square(cumulants.mean) - \
        #                                 be.square(self.params.loc) - 2.0*scale)

        return [ParamsGaussian(d_ulogZ, d_lnslogZ)]


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
        self.set_params([ParamsGaussian(self.moments.mean, be.log(self.moments.var))])

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
        be.mix_(be.float_scalar(1-shrinkage), var, be.ones_like(var))
        self.set_params([ParamsGaussian(self.params.loc, be.log(var))])

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
        if not self.center:
            return be.divide(scale, observations)
        return be.divide(scale, be.subtract(self.get_center(), observations))

    def rescale_cumulants(self, cumulants):
        """
        Rescales the cumulants associated with the layer.  The variance is doubly-rescaled.

        Args:
            cumulants (CumulantsTAP)

        Returns:
            rescaled cumulants (CumulantsTAP)
        """
        return CumulantsTAP(self.rescale(cumulants.mean),
                            self.rescale(self.rescale(cumulants.variance)))

    def reciprocal_scale(self):
        """
        Returns a tensor of shape (num_units) providing a reciprocal scale for each unit

        Args:
            None
        Returns:
            scale (tensor)
        """
        return be.reciprocal(be.exp(self.params.log_var))

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
            derivs (List[namedtuple]): List[param_name: tensor] (contains gradient)

        """
        # initialize tensors for the location and scale derivatives
        loc = be.zeros_like(self.params[0]),
        log_var = be.zeros_like(self.params[1])

        # compute the derivative with respect to the location parameter
        diff_scaled = self.rescale(self.params.loc - units)
        loc = be.mean(weighting_function(diff_scaled),
                    axis=0)

        if penalize:
            loc = self.get_penalty_grad(loc, 'loc')

        # compute the derivative with respect to the scale parameter
        log_var = -0.5 * be.mean(weighting_function(be.square(be.subtract(
                                    self.params.loc, units))), axis=0)
        for i in range(len(connected_units)):
            log_var += be.batch_quadratic(connected_units[i], connected_weights[i],
                            weighting_function(units), axis=0) / len(units)

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
        mean, _ = self.conditional_params(scaled_units, weights, beta)
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
        mean, _ = self.conditional_params(scaled_units, weights, beta)
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
        mean, var = self.conditional_params(scaled_units, weights, beta)
        r = self.rand(be.shape(mean))
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

    def envelope_random(self, array_or_shape):
        """
        Generate a random sample with the same type as the layer.
        For a Gaussian layer, draws from a normal distribution
        with the mean and variance from the moments.

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

        mean = self.moments.mean
        var = self.moments.var
        r = self.rand(shape)

        return be.add(mean, be.multiply(be.sqrt(var), r))
