# -*- coding: utf-8 -*-
"""
This module defines classes that represent the state of some model fit metric,
derived from summary information about the current state of the model
(encapsulated in MetricState).

"""

from collections import namedtuple
import math

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
        self.mean_square_error = 0
        self.norm = 0

    def reset(self) -> None:
        """
        Reset the metric to it's initial state.

        Notes:
            Changes norm and mean_square_error in place.

        Args:
            None

        Returns:
            None

        """
        self.mean_square_error = 0
        self.norm = 0

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the reconstruction error using a batch
        of observations and a batch of reconstructions.

        Notes:
            Changes norm and mean_square_error in place.

        Args:
            update_args: uses visible layer of minibatch and reconstructions


        Returns:
            None

        """
        self.norm += len(update_args.minibatch.units[0])
        self.mean_square_error += be.tsum(
            (update_args.minibatch.units[0] -
             update_args.reconstructions.units[0])**2)

    def value(self) -> float:
        """
        Get the value of the reconstruction error.

        Args:
            None

        Returns:
            reconstruction error (float)

        """
        if self.norm:
            return math.sqrt(self.mean_square_error / self.norm)
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

    def __init__(self, downsample=100):
        """
        Create EnergyDistance object.

        Args:
            downsample (int; optional): how many samples to use

        Returns:
            energy distance object

        """
        self.energy_distance = 0
        self.norm = 0
        self.downsample = 100

    def reset(self) -> None:
        """
        Reset the metric to it's initial state.

        Note:
            Modifies norm and energy_distance in place.

        Args:
            None

        Returns:
            None

        """
        self.energy_distance = 0
        self.norm = 0

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the energy distance using a batch
        of observations and a batch of fantasy particles.

        Notes:
            Changes norm and energy_distance in place.

        Args:
            update_args: uses visible layer of minibatch and samples

        Returns:
            None

        """
        self.norm += 1
        self.energy_distance += \
            be.fast_energy_distance(update_args.minibatch.units[0],
                                    update_args.samples.units[0],
                                    self.downsample)

    def value(self) -> float:
        """
        Get the value of the energy distance.

        Args:
            None

        Returns:
            energy distance (float)

        """
        if self.norm:
            return self.energy_distance / self.norm
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
        self.energy_gap = 0
        self.norm = 0

    def reset(self) -> None:
        """
        Reset the metric to it's initial state.

        Note:
            Modifies norm and energy_gap in place.

        Args:
            None

        Returns:
            None

        """
        self.energy_gap = 0
        self.norm = 0

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the energy gap using a batch
        of observations and a batch of fantasy particles.

        Notes:
            Changes norm and energy_gap in place.

        Args:
            update_args: uses all layers of minibatch and random_samples, and model

        Returns:
            None

        """
        self.norm += 1
        self.energy_gap += be.mean(update_args.model
                                   .joint_energy(update_args.minibatch))
        self.energy_gap -= be.mean(update_args.model
                                   .joint_energy(update_args.random_samples))

    def value(self):
        """
        Get the value of the energy gap.

        Args:
            None

        Returns:
            energy gap (float)

        """
        if self.norm:
            return self.energy_gap / self.norm
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
            energy z-score object

        """
        self.data_mean = 0
        self.random_mean = 0
        self.random_mean_square = 0

    def reset(self) -> None:
        """
        Reset the metric to it's initial state.

        Note:
            Modifies norm, random_mean, and random_mean_square in place.

        Args:
            None

        Returns:
            None

        """
        self.data_mean = 0
        self.random_mean = 0
        self.random_mean_square = 0

    def update(self, update_args: MetricState) -> None:
        """
        Update the estimate for the energy z-score using a batch
        of observations and a batch of fantasy particles.

        Notes:
            Changes norm, random_mean, and random_mean_square in place.

        Args:
            update_args: uses all layers of minibatch and random_samples, and model

        Returns:
            None

        """
        self.data_mean += be.mean(update_args.model
                                  .joint_energy(update_args.minibatch))
        self.random_mean += be.mean(update_args.model
                                     .joint_energy(update_args.random_samples))
        self.random_mean_square += be.mean(update_args.model
                                           .joint_energy(update_args.random_samples)**2)

    def value(self) -> float:
        """
        Get the value of the energy z-score.

        Args:
            None

        Returns:
            energy z-score (float)

        """
        if self.random_mean_square:
            return (self.data_mean - self.random_mean) / math.sqrt(self.random_mean_square)
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
        self.calc.calculate(energy)

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
