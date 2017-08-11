# -*- coding: utf-8 -*-
"""
This module defines classes that represent the state of some model fit metric,
derived from summary information about the current state of the model
(encapsulated in MetricState).

"""

from collections import namedtuple
import numpy as np

from . import math_utils
from . import backends as be

# ----- CLASSES ----- #

"""
A namedtuple of states.

Attributes:
    minibatch: a State object with the visible units clamped to the observations
    reconstructions: a State object with one update from the observations
    random_samples: a State object with random units for all layers
    samples: a State object sampled from the model (i.e., fantasy particles)
    model: a Model object

"""
MetricState = namedtuple('MetricState', [
    'minibatch',
    'reconstructions',
    'random_samples',
    'samples',
    'model'
])

class ReconstructionError(object):
    """
    Compute the root-mean-squared error between observations and their
    reconstructions using minibatches.

    """

    name = 'ReconstructionError'

    def __init__(self):
        """
        Create a ReconstructionError object.

        Args:
            None

        Returns:
            ReconstructionERror

        """
        self.calc = math_utils.MeanCalculator()

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the reconstruction error using a batch
        of observations and a batch of reconstructions.

        Args:
            update_args: uses visible layer of minibatch and reconstructions

        Returns:
            None

        """
        mse = be.square(
                be.subtract(update_args.reconstructions.units[0],
                            update_args.minibatch.units[0])
                )
        self.calc.update(be.tsum(mse, axis=1))

    def value(self) -> float:
        """
        Get the value of the reconstruction error.

        Args:
            None

        Returns:
            reconstruction error (float)

        """
        if self.calc.num is not None:
            return np.sqrt(self.calc.mean)
        else:
            return None


class EnergyDistance(object):
    """
    Compute the energy distance between two distributions using
    minibatches of sampled configurations.

    Szekely, G.J. (2002)
    E-statistics: The Energy of Statistical Samples.
    Technical Report BGSU No 02-16.

    """

    name = 'EnergyDistance'

    def __init__(self):
        """
        Create EnergyDistance object.

        Args:
            downsample (int; optional): how many samples to use

        Returns:
            energy distance object

        """
        self.calc = math_utils.MeanCalculator()

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the energy distance using a batch
        of observations and a batch of fantasy particles.

        Args:
            update_args: uses visible layer of minibatch and samples

        Returns:
            None

        """
        energy_distance = be.energy_distance(update_args.minibatch.units[0],
                                                  update_args.samples.units[0])
        self.calc.update(be.float_tensor(np.array([energy_distance])))

    def value(self) -> float:
        """
        Get the value of the energy distance.

        Args:
            None

        Returns:
            energy distance (float)

        """
        if self.calc.num is not None:
            return self.calc.mean
        else:
            return None


class EnergyGap(object):
    """
    Samples drawn from a model should have much lower energy
    than purely random samples. The "energy gap" is the average
    energy difference between samples from the model and random
    samples.

    """

    name = 'EnergyGap'

    def __init__(self):
        """
        Create an EnergyGap object.

        Args:
            None

        Returns:
            energy gap object

        """
        self.calc = math_utils.MeanCalculator()

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc.reset()

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the energy gap using a batch
        of observations and a batch of fantasy particles.

        Args:
            update_args: uses all layers of minibatch and random_samples, and model

        Returns:
            None

        """
        energy_data = update_args.model.joint_energy(update_args.minibatch)
        energy_random = update_args.model.joint_energy(update_args.random_samples)
        self.calc.update(energy_data)
        self.calc.update(-energy_random)

    def value(self):
        """
        Get the value of the energy gap.

        Args:
            None

        Returns:
            energy gap (float)

        """
        if self.calc.num is not None:
            # double the mean from double counting the data and random sets
            return 2*self.calc.mean
        else:
            return None


class EnergyZscore(object):
    """
    Samples drawn from a model should have much lower energy
    than purely random samples. The "energy gap" is the average
    energy difference between samples from the model and random
    samples. The "energy z-score" is the energy gap divided by
    the standard deviation of the energy taken over random
    samples.

    """

    name = 'EnergyZscore'

    def __init__(self):
        """
        Create an EnergyZscore object.

        Args:
            None

        Returns:
            EnergyZscore object

        """
        self.calc_data = math_utils.MeanCalculator()
        self.calc_random = math_utils.MeanVarianceCalculator()

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.calc_data.reset()
        self.calc_random.reset()

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the energy z-score using a batch
        of observations and a batch of fantasy particles.

        Args:
            update_args: uses all layers of minibatch and random_samples, and model

        Returns:
            None

        """
        energy_data = update_args.model.joint_energy(update_args.minibatch)
        energy_random = update_args.model.joint_energy(update_args.random_samples)
        self.calc_data.update(energy_data)
        self.calc_random.update(energy_random)

    def value(self) -> float:
        """
        Get the value of the energy z-score.

        Args:
            None

        Returns:
            energy z-score (float)

        """
        if self.calc_data.num is not None:
            z = (self.calc_data.mean - self.calc_random.mean) / np.sqrt(self.calc_random.var)
            return z
        else:
            return None

class HeatCapacity(object):
    """
    Compute the heat capacity of the system thought of as a spin system.

    We take the HC to be the second cumulant of the energy, or alternately
    the negative second derivative with respect to inverse temperature of
    the Gibbs free energy.  In order to estimate this quantity we perform
    Gibbs sampling starting from random samples drawn from the visible layer's
    distribution.

    """

    name = 'HeatCapacity'

    def __init__(self):
        """
        Create HeatCapacity object.

        Args:
            None

        Returns:
            None

        """
        self.calc = math_utils.MeanVarianceCalculator()

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            The HeatCapacity object.

        """
        self.calc.reset()

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the heat capacity.

        Args:
            update_args: uses all layers of random_samples, and model

        Returns:
            None

        """
        energy = update_args.model.joint_energy(update_args.samples)
        self.calc.update(energy)

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
        else:
            return None

class WeightSparsity(object):
    """
    Compute the weight sparsity of the model as the formula

    p = \sum_j(\sum_i w_ij^2)^2/\sum_i w_ij^4

    Tubiana, J., Monasson, R. (2017)
    Emergence of Compositional Representations in Restricted Boltzmann Machines,
    PRL 118, 138301 (2017)

    """

    name = 'WeightSparsity'

    def __init__(self):
        """
        Create WeightSparsity object.

        Args:
            None
        Returns:
            WeightSparsity object

        """
        self.p = 0

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.p = None

    def update(self, update_args: MetricState) -> None:
        """
        Compute the weight sparsity of the model

        Args:
            update_args: uses model only

        Returns:
            None

        """
        # TODO: should this use the weights of all of the layers?
        w = update_args.model.weights[0].W()
        (n,m) = be.shape(w)
        w2 = be.tsum(be.square(w), axis=0)
        w4 = be.tsum(be.square(w2), axis=0)
        self.p = 1.0/float(n*m) * be.tsum(be.square(w2)/be.clip(w4, a_min=be.EPSILON))

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
        else:
            return None

class WeightSquare(object):
    """
    Compute the mean squared weights of the model per hidden unit

    w2 = 1/(#hidden units)*\sum_ij w_ij^2

    Tubiana, J., Monasson, R. (2017)
    Emergence of Compositional Representations in Restricted Boltzmann Machines,
    PRL 118, 138301 (2017)

    """

    name = 'WeightSquare'

    def __init__(self):
        """
        Create WeightSquare object.

        Args:
            None
        Returns:
            WeightSquare object

        """
        self.mw2 = 0

    def reset(self) -> None:
        """
        Reset the metric to its initial state.

        Args:
            None

        Returns:
            None

        """
        self.mw2 = None

    def update(self, update_args: MetricState) -> None:
        """
        Compute the weight square of the model

        Args:
            update_args: uses model only

        Returns:
            None

        """
        # TODO: should this use the weights of all of the layers?
        w = update_args.model.weights[0].W()
        (n,m) = be.shape(w)
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
        else:
            return None
