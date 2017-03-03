import math
from numba import jit
from . import backends as be

# ----- CLASSES ----- #


class ReconstructionError(object):

    name = 'ReconstructionError'

    def __init__(self):
        self.mean_square_error = 0
        self.norm = 0

    def reset(self):
        self.mean_square_error = 0
        self.norm = 0

    def update(self, minibatch=None, reconstructions=None, **kwargs):
        self.norm += len(minibatch)
        self.mean_square_error += be.tsum((minibatch - reconstructions)**2)

    def value(self):
        if self.norm:
            return math.sqrt(self.mean_square_error / self.norm)
        else:
            return None


class EnergyDistance(object):

    name = 'EnergyDistance'

    def __init__(self, downsample=100):
        self.energy_distance = 0
        self.norm = 0
        self.downsample = 100

    def reset(self):
        self.energy_distance = 0
        self.norm = 0

    def update(self, minibatch=None, samples=None, **kwargs):
        self.norm += 1
        self.energy_distance += be.fast_energy_distance(minibatch, samples,
                                                     self.downsample)

    def value(self):
        if self.norm:
            return self.energy_distance / self.norm
        else:
            return None


class EnergyGap(object):

    name = 'EnergyGap'

    def __init__(self):
        self.energy_gap = 0
        self.norm = 0

    def reset(self):
        self.energy_gap = 0
        self.norm = 0

    def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs):
        self.norm += 1
        self.energy_gap += be.mean(amodel.marginal_free_energy(minibatch))
        self.energy_gap -= be.mean(amodel.marginal_free_energy(random_samples))

    def value(self):
        if self.norm:
            return self.energy_gap / self.norm
        else:
            return None


class EnergyZscore(object):

    name = 'EnergyZscore'

    def __init__(self):
        self.data_mean = 0
        self.random_mean = 0
        self.random_mean_square = 0

    def reset(self):
        self.data_mean = 0
        self.random_mean = 0
        self.random_mean_square = 0

    def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs):
        self.data_mean += be.mean(amodel.marginal_free_energy(minibatch))
        self.random_mean +=  be.mean(amodel.marginal_free_energy(random_samples))
        self.random_mean_square +=  be.mean(amodel.marginal_free_energy(random_samples)**2)

    def value(self):
        if self.random_mean_square:
            return (self.data_mean - self.random_mean) / math.sqrt(self.random_mean_square)
        else:
            return None
