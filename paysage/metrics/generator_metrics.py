from math import sqrt, log

from .. import math_utils
from .. import backends as be

# ----- CLASSES ----- #

class ReconstructionError(object):
    """
    Compute the root-mean-squared error between observations and their
    reconstructions using minibatches, rescaled by the minibatch variance.

    """
    def __init__(self, name='ReconstructionError'):
        """
        Create a ReconstructionError object.

        Args:
            name (str; optional): metric name

        Returns:
            ReconstructionError

        """
        self.calc = math_utils.MeanCalculator()
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def update(self, assessment) -> None:
        """
        Update the estimate for the reconstruction error using a batch
        of observations and a batch of reconstructions.

        Args:
            assessment (ModelAssessment): uses data_state and reconstructions

        Returns:
            None

        """
        var = be.EPSILON + be.var(assessment.data_state.get_visible())
        rec = assessment.reconstructions.get_visible()
        state = assessment.data_state.get_visible()
        mse = be.mean(be.square(be.subtract(rec, state)), axis=1) / var
        self.calc.update(mse)

    def value(self) -> float:
        """
        Get the value of the reconstruction error.

        Args:
            None

        Returns:
            reconstruction error (float)

        """
        if self.calc.num is not None:
            return sqrt(self.calc.mean)
        return None


class EnergyCoefficient(object):
    """
    Compute a normalized energy distance between two distributions using
    minibatches of sampled configurations.

    Szekely, G.J. (2002)
    E-statistics: The Energy of Statistical Samples.
    Technical Report BGSU No 02-16.

    """
    def __init__(self, name='EnergyCoefficient'):
        """
        Create EnergyCoefficient object.

        Args:
            None

        Returns:
            EnergyCoefficient object

        """
        self.calc = math_utils.MeanCalculator()
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def _energy_coefficient(self, x, y):
        """
        Compute the energy coefficient.

        Args:
            x (tensor ~ (num_samples_x, num_units))
            y (tensor ~ (num_samples_y, num_units))

        Returns:
            float

        """
        d1 = be.mean(math_utils.pdist(x, y))
        d2 = be.mean(math_utils.pdist(x, x))
        d3 = be.mean(math_utils.pdist(y, y))
        return sqrt(max(0, (2*d1 - d2 - d3) / max(2*d1, be.EPSILON)))

    def update(self, assessment) -> None:
        """
        Update the estimate for the energy coefficient using a batch
        of observations and a batch of fantasy particles.

        Args:
            assessment (ModelAssessment): uses data_state and model_state

        Returns:
            None

        """
        ecoeff = self._energy_coefficient(assessment.data_state[0],
                                          assessment.model_state[0])
        self.calc.update(be.float_tensor([ecoeff]))

    def value(self) -> float:
        """
        Get the value of the energy coefficient.

        Args:
            None

        Returns:
            energy coefficient (float)

        """
        if self.calc.num is not None:
            return self.calc.mean
        return None


class KLDivergence(object):
    """
    Compute the KL divergence between two samples using the method of:

    "Divergence Estimation for Multidimensional Densities Via k-Nearest Neighbor
    Distances"
    by Qing Wang, Sanjeev R. Kulkarni and Sergio Verdú

    KL(P || Q) = \int dx p(x) log(p(x)/q(x))

    p ~ data samples
    q ~ model samples

    We provide the option to remove dependence on dimension, true by default.

    """
    def __init__(self, k=5, name='KLDivergence', divide_dimension=True):
        """
        Create KLDivergence object.

        Args:
            k (int; optional): which nearest neighbor to use
            name (str; optional): metric name
            divide_dimension (bool; optional): whether to divide the divergence
                by the number of dimensions

        Returns:
            KLDivergence object

        """
        self.calc = math_utils.MeanCalculator()
        self.k = k
        self.name = name
        self.divide_dim = divide_dimension

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    @classmethod
    def klpq(cls, x, y, k, divide_dim):
        """
        Compute the forward KL divergence.

        Args:
            x (tensor ~ (num_samples_x, num_units))
            y (tensor ~ (num_samples_y, num_units))
            k (int)

        Returns:
            float

        """
        n = len(x)
        m = len(y)

        _, x_dist = math_utils.find_k_nearest_neighbors(x, x, k+1)
        _, y_dist = math_utils.find_k_nearest_neighbors(x, y, k)

        be.clip_(x_dist, a_min = be.EPSILON)
        be.clip_(y_dist, a_min = be.EPSILON)

        if divide_dim:
            d = 1.0
        else:
            d = be.shape(x)[1] # the dimension of the space

        return d*be.tsum(be.log(y_dist / x_dist))/n + log(m/(n-1))

    def update(self, assessment) -> None:
        """
        Update the estimate for the KL divergence using a batch
        of observations and a batch of fantasy particles.

        Args:
            assessment (ModelAssessment): uses data_state and model_state

        Returns:
            None

        """
        klpq = self.klpq(assessment.data_state[0], assessment.model_state[0],
                         self.k, self.divide_dim)
        self.calc.update(be.float_tensor([klpq]))

    def value(self) -> float:
        """
        Get the value of the KL divergence estimation.

        Args:
            None

        Returns:
            KL divergence estimation (float)

        """
        if self.calc.num is not None:
            return self.calc.mean
        return None


class ReverseKLDivergence(object):
    """
    Compute the reverse KL divergence between two samples using the method of:

    "Divergence Estimation for Multidimensional Densities Via k-Nearest Neighbor
    Distances"
    by Qing Wang, Sanjeev R. Kulkarni and Sergio Verdú

    KL(P || Q) = \int dx p(x) log(p(x)/q(x))

    p ~ model samples
    q ~ data samples

    We provide the option to divide out the dimension.
    """
    def __init__(self, k=5, name='ReverseKLDivergence', divide_dimension=True):
        """
        Create ReverseKLDivergence object.

        Args:
            k (int; optional): which nearest neighbor to use
            name (str; optional): metric name
            divide_dimension (bool; optional): whether to divide the divergence
                by the number of dimensions

        Returns:
            ReverseKLDivergence object

        """
        self.calc = math_utils.MeanCalculator()
        self.k = k
        self.name = name
        self.divide_dim = divide_dimension

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    @classmethod
    def klqp(cls, y, x, k, divide_dim):
        """
        Compute the KL divergence.

        Args:
            y (tensor ~ (num_samples_y, num_units))
            x (tensor ~ (num_samples_x, num_units))

        Returns:
            float

        """
        n = len(x)
        m = len(y)

        _, x_dist = math_utils.find_k_nearest_neighbors(x, x, k+1)
        _, y_dist = math_utils.find_k_nearest_neighbors(x, y, k)

        be.clip_(x_dist, a_min = be.EPSILON)
        be.clip_(y_dist, a_min = be.EPSILON)

        if divide_dim:
            d = 1.0
        else:
            d = be.shape(x)[1] # the dimension of the space

        return d*be.tsum(be.log(y_dist / x_dist))/n + log(m/(n-1))


    def update(self, assessment) -> None:
        """
        Update the estimate for the reverse KL divergence using a batch
        of observations and a batch of fantasy particles.

        Args:
            assessment (ModelAssessment): uses data_state and model_state

        Returns:
            None

        """
        klqp = self.klqp(assessment.data_state[0], assessment.model_state[0],
                        self.k, self.divide_dim)
        self.calc.update(be.float_tensor([klqp]))

    def value(self) -> float:
        """
        Get the value of the reverse KL divergence estimate.

        Args:
            None

        Returns:
            reverse KL divergence estimate (float)

        """
        if self.calc.num is not None:
            return self.calc.mean
        return None


class JensenShannonDivergence(object):
    """
    Compute the JS divergence between two samples using the method of:

    "Divergence Estimation for Multidimensional Densities Via k-Nearest Neighbor
    Distances"
    by Qing Wang, Sanjeev R. Kulkarni and Sergio Verdú

    JS(P || Q) = 1/2*KL(P || 1/2(P + Q)) + 1/2*KL(Q || 1/2(P + Q))

    p ~ model samples
    q ~ data samples

    We provide the option to divide out by the dimension of the dataset.
    """
    def __init__(self, k=5, name='JensenShannonDivergence', divide_dimension=True):
        """
        Create JensenShannonKLDivergence object.

        Args:
            k (int; optional): which nearest neighbor to use
            name (str; optional): metric name
            divide_dimension (bool; optional): whether to divide the divergence
                by the number of dimensions

        Returns:
            JensenShannonDivergence object

        """
        self.calc = math_utils.MeanCalculator()
        self.k = k
        self.name = name
        self.divide_dim = divide_dimension

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def _js(self, x, y):
        """
        Compute the Jensen-Shannon divergence.

        Args:
            x (tensor ~ (num_samples_x, num_units))
            y (tensor ~ (num_samples_y, num_units))

        Returns:
            float

        """
        js = 0

        n = len(x)
        m = len(y)
        if self.divide_dim:
            d = 1.0
        else:
            d = be.shape(x)[1] # the dimension of the space

        _, x_dist = math_utils.find_k_nearest_neighbors(x, x, self.k+1)
        _, y_dist = math_utils.find_k_nearest_neighbors(x, y, self.k)

        be.clip_(x_dist, a_min = be.EPSILON)
        be.clip_(y_dist, a_min = be.EPSILON)

        r = x_dist / y_dist

        js += log(2) - \
             be.tsum(be.logaddexp(be.zeros_like(r), log((n-1)/m) + d*be.log(r)))/n

        n = len(y)
        m = len(x)
        _, x_dist = math_utils.find_k_nearest_neighbors(y, y, self.k+1)
        _, y_dist = math_utils.find_k_nearest_neighbors(y, x, self.k)

        be.clip_(x_dist, a_min = be.EPSILON)
        be.clip_(y_dist, a_min = be.EPSILON)

        r = x_dist / y_dist

        js += log(2) - \
             be.tsum(be.logaddexp(be.zeros_like(r), log((n-1)/m) + d*be.log(r)))/n

        return 0.5*js

    def update(self, assessment) -> None:
        """
        Update the estimate for the JS divergence using a batch
        of observations and a batch of fantasy particles.

        Args:
            assessment (ModelAssessment): uses data_state and model_state

        Returns:
            None

        """
        js = self._js(assessment.data_state[0], assessment.model_state[0])
        self.calc.update(be.float_tensor([js]))

    def value(self) -> float:
        """
        Get the value of the reverse JS divergence estimate.

        Args:
            None

        Returns:
            JS divergence estimate (float)

        """
        if self.calc.num is not None:
            return self.calc.mean
        return None


class FrechetScore(object):
    """
    Compute the Frechet Score between two samples. Based on an idea from:

    "GANs Trained by a Two Time-Scale Update Rule Converge to a
    Local Nash Equilibrium"
    by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler
    Sepp Hochreiter

    but without the inception network.

    """
    def __init__(self, name='FrechetScore'):
        """
        Create FrechetScore object.

        Args:
            None

        Returns:
            FrechetScore object

        """
        self.calc = math_utils.MeanCalculator()
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def _fid(self, x, y):
        """
        Compute the Frechet Score.

        Args:
            x (tensor ~ (num_samples_x, num_units)): data
            y (tensor ~ (num_samples_y, num_units)): fantasy

        Returns:
            float

        """
        m1 = be.mean(x, axis=0)
        m2 = be.mean(y, axis=0)

        C1 = be.cov(x, x)
        C2 = be.cov(y, y)

        result = be.tsum(be.square(m1 - m2))
        result += be.tsum(be.diag(C1))
        result += be.tsum(be.diag(C2))

        tmp = be.matrix_sqrt(be.dot(C1, C2))
        result -= 2 * be.tsum(be.diag(tmp))

        return result

    def update(self, assessment) -> None:
        """
        Update the estimate for the Frechet Score using a batch
        of observations and a batch of fantasy particles.

        Args:
            assessment (ModelAssessment): uses data_state and model_state

        Returns:
            None

        """
        fid = self._fid(assessment.data_state[0], assessment.model_state[0])
        self.calc.update(be.float_tensor(fid))

    def value(self) -> float:
        """
        Get the value of the Frechet Score estimate.

        Args:
            None

        Returns:
            Frechet Score estimate (float)

        """
        if self.calc.num is not None:
            return self.calc.mean
        return None


class HeatCapacity(object):
    """
    Compute the heat capacity of the model per parameter.

    We take the HC to be the second cumulant of the energy, or alternately
    the negative second derivative with respect to inverse temperature of
    the Gibbs free energy.  In order to estimate this quantity we perform
    Gibbs sampling starting from random samples drawn from the visible layer's
    distribution.  This is rescaled by the number of units parameters in the model.

    """
    def __init__(self, name='HeatCapacity'):
        """
        Create HeatCapacity object.

        Args:
            None

        Returns:
            None

        """
        self.calc = math_utils.MeanVarianceCalculator()
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def update(self, assessment) -> None:
        """
        Update the estimate for the heat capacity.

        Args:
            assessment (ModelAssessment): uses model and model_state

        Returns:
            None

        """
        energy = assessment.model.joint_energy(assessment.model_state)
        self.num_params = assessment.model.num_parameters()
        self.calc.update(energy / sqrt(sqrt(assessment.model.num_parameters())))

    def value(self) -> float:
        """
        Get the value of the heat capacity.

        Args:
            None

        Returns:
            heat capacity (float)

        """
        if self.calc.num:
            return self.calc.var
        return None


class WeightSparsity(object):
    """
    Compute the weight sparsity of the model as the formula

    p = \sum_j(\sum_i w_ij^2)^2/\sum_i w_ij^4

    Tubiana, J., Monasson, R. (2017)
    Emergence of Compositional Representations in Restricted Boltzmann Machines,
    PRL 118, 138301 (2017)

    """
    def __init__(self, name='WeightSparsity'):
        """
        Create WeightSparsity object.

        Args:
            None
        Returns:
            WeightSparsity object

        """
        self.p = None
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.p = None

    def update(self, assessment) -> None:
        """
        Compute the weight sparsity of the model

        Notes:
            If the value already exists, it is not updated.
            Call reset() between model updates.

        Args:
            assessment (ModelAssessment): uses model

        Returns:
            None

        """
        if self.p is not None:
            return
        # TODO: should this use the weights of all of the layers?
        w = assessment.model.connections[0].weights.W()
        (n,m) = be.shape(w)
        w2 = be.square(w)
        w4 = be.square(w2)
        self.p = 1.0/float(n*m) * be.tsum(be.square(be.tsum(w2, axis=0))/ be.tsum(w4, axis=0))

    def value(self) -> float:
        """
        Get the value of the weight sparsity.

        Args:
            None

        Returns:
            weight sparsity (float)

        """
        if self.p is not None:
            return self.p
        return None


class WeightSquare(object):
    """
    Compute the mean squared weights of the model per hidden unit

    w2 = 1/(#hidden units)*\sum_ij w_ij^2

    Tubiana, J., Monasson, R. (2017)
    Emergence of Compositional Representations in Restricted Boltzmann Machines,
    PRL 118, 138301 (2017)

    """
    def __init__(self, name='WeightSquare'):
        """
        Create WeightSquare object.

        Args:
            None
        Returns:
            WeightSquare object

        """
        self.mw2 = None
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.mw2 = None

    def update(self, assessment) -> None:
        """
        Compute the weight square of the model.

        Notes:
            If the value already exists, it is not updated.
            Call reset() between model updates.

        Args:
            assessment (ModelAssessment): uses model

        Returns:
            None

        """
        if self.mw2 is not None:
            return
        # TODO: should this use the weights of all of the layers?
        w = assessment.model.connections[0].weights.W()
        (_,m) = be.shape(w)
        w2 = be.square(w)
        self.mw2 = 1.0/float(m) * be.tsum(w2)

    def value(self) -> float:
        """
        Get the value of the weight sparsity.

        Args:
            None

        Returns:
            weight sparsity (float)

        """
        if self.mw2 is not None:
            return self.mw2
        return None


class TAPFreeEnergy(object):
    """
    Compute the TAP2 free energy of the model seeded from some number of
    random magnetizations.  This value approximates -lnZ_model

    """
    def __init__(self, num_samples=2, name='TAPFreeEnergy'):
        """
        Create TAPFreeEnergy object.

        Args:
            num_samples (int): number of samples to average over

        Returns:
            None

        """
        self.calc = math_utils.MeanVarianceCalculator()
        self.num_samples = num_samples
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None
        """
        self.calc.reset()

    def update(self, assessment) -> None:
        """
        Update the estimate for the TAP free energy.

        Args:
            assessment (ModelAssessment): uses model

        Returns:
            None

        """
        for _ in range(self.num_samples):
            _, fe = assessment.model.compute_StateTAP()
            self.calc.update(be.float_tensor([fe]))

    def value(self) -> float:
        """
        Get the average TAP free energy.

        Args:
            None

        Returns:
            the average TAP free energy (float)

        """
        if self.calc.num:
            return self.calc.mean
        return None


class TAPLogLikelihood(object):
    """
    Compute the log likelihood of the data using the TAP2 approximation of -lnZ_model
    """
    def __init__(self, num_samples=2, name='TAPLogLikelihood'):
        """
        Create TAPLogLikelihood object.

        Args:
            num_samples (int): number of samples to average over

        Returns:
            None

        """
        self.calc = math_utils.MeanVarianceCalculator()
        self.num_samples = num_samples
        self.name = name

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None
        """
        self.calc.reset()

    def update(self, assessment) -> None:
        """
        Update the estimate for the TAP free energy and the marginal free energy
         (actually the average per sample)

        Args:
            assessment (ModelAssessment): uses model

        Returns:
            None

        """
        state = assessment.data_state
        rbm = assessment.model
        stepsize = be.shape(state[0])[0]
        for _ in range(self.num_samples):
            _, TAP_fe = rbm.compute_StateTAP()
            vis = -be.tsum(rbm.layers[0].energy(state[0]))
            c_params = rbm.layers[1].conditional_params(
                rbm._connected_rescaled_units(1, state),
                rbm._connected_weights(1))
            marginal_fe = vis + be.tsum(
                 rbm.layers[1].log_partition_function(c_params, be.zeros_like(c_params)))
            self.calc.update(be.float_tensor([TAP_fe + marginal_fe/stepsize]))

    def value(self) -> float:
        """
        Get the average TAP log likelihood.

        Args:
            None

        Returns:
            the average TAP log likelihood (float)

        """
        if self.calc.num:
            return self.calc.mean
        return None
