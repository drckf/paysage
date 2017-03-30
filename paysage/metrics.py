import math
from . import backends as be

# ----- CLASSES ----- #


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

    def reset(self):
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

    #TODO: use State objects instead of tensors
    def update(self, minibatch=None, reconstructions=None, **kwargs):
        """
        Update the estimate for the reconstruction error using a batch
        of observations and a batch of reconstructions.

        Notes:
            Changes norm and mean_square_error in place.

        Args:
            minibatch (tensor (num_samples, num_units))
            reconstructions (tensor (num_samples, num))
            kwargs: key word arguments
                not used, but helpful for looping through metric functions

        Returns:
            None

        """
        self.norm += len(minibatch)
        self.mean_square_error += be.tsum((minibatch - reconstructions)**2)

    def value(self):
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

    def reset(self):
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

    #TODO: use State objects instead of tensors
    def update(self, minibatch=None, samples=None, **kwargs):
        """
        Update the estimate for the energy distance using a batch
        of observations and a batch of fantasy particles.

        Notes:
            Changes norm and energy_distance in place.

        Args:
            minibatch (tensor (num_samples, num_units))
            samples (tensor (num_samples, num)): fantasy particles
            kwargs: key word arguments
                not used, but helpful for looping through metric functions

        Returns:
            None

        """
        self.norm += 1
        self.energy_distance += be.fast_energy_distance(minibatch, samples,
                                                     self.downsample)

    def value(self):
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

    def reset(self):
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

    #TODO: use State objects instead of tensors
    def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs):
        """
        Update the estimate for the energy gap using a batch
        of observations and a batch of fantasy particles.

        Notes:
            Changes norm and energy_gap in place.

        Args:
            minibatch (tensor (num_samples, num_units)):
                samples from the model
            random_samples (tensor (num_samples, num))
            amodel (Model): the model
            kwargs: key word arguments
                not used, but helpful for looping through metric functions

        Returns:
            None

        """
        self.norm += 1
        self.energy_gap += be.mean(amodel.marginal_free_energy(minibatch))
        self.energy_gap -= be.mean(amodel.marginal_free_energy(random_samples))

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

    def reset(self):
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

    #TODO: use State objects instead of tensors
    def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs):
        """
        Update the estimate for the energy z-score using a batch
        of observations and a batch of fantasy particles.

        Notes:
            Changes norm, random_mean, and random_mean_square in place.

        Args:
            minibatch (tensor (num_samples, num_units)):
                samples from the model
            random_samples (tensor (num_samples, num))
            amodel (Model): the model
            kwargs: key word arguments
                not used, but helpful for looping through metric functions

        Returns:
            None

        """
        self.data_mean += be.mean(amodel.marginal_free_energy(minibatch))
        self.random_mean +=  be.mean(amodel.marginal_free_energy(random_samples))
        self.random_mean_square +=  be.mean(amodel.marginal_free_energy(random_samples)**2)

    def value(self):
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
