import numpy
from .backends import numba_engine as en

LOG2 = 0.6931471805599453

#----- LAYER CLASSES -----#

class GaussianLayer(object):
    
    constraints = []
    
    def __init__(self):
        pass
        
    def prox(self, vis):
        return vis
        
    def mean(self, loc):
        return loc
        
    def inverse_mean(self, mean):
        return mean
        
    def log_partition_function(self, scale):
        return -numpy.log(scale)
    
    def sample_state(self, loc, scale):
        return loc + scale * numpy.random.normal(loc=0.0, scale=1.0, size=loc.shape)
        
    def random(self, loc, scale):
        return numpy.random.normal(loc=0.0, scale=1.0, size=loc.shape)


class IsingLayer(object):
    
    constraints = []

    def __init__(self):
        pass
        
    def prox(self, vis):
        return 2.0 * (vis > 0.0).astype(numpy.float32) - 1.0
        
    def mean(self, loc):
        return numpy.tanh(loc)
        
    def inverse_mean(self, mean):
        return numpy.arctanh(loc)
        
    def log_partition_function(self, loc):
        return -LOG2 + numpy.logaddexp(-loc, loc)
        
    def sample_state(self, loc):
        return en.random_ising(en.expit(loc))
        
    def random(self, loc):
        return 2.0 * numpy.random.randint(0, 2, loc.shape).astype(numpy.float32) - 1.0
        
        
class BernoulliLayer(object):
    
    constraints = []
    
    def __init__(self):
        pass
        
    def prox(self, vis):
        return (vis > 0.0).astype(numpy.float32)
        
    def mean(self, loc):
        return en.expit(loc)
        
    def inverse_mean(self, mean):
        return numpy.log(mean / (1 - mean))
        
    def log_partition_function(self, loc):
        return numpy.logaddexp(0, loc)
        
    def sample_state(self, loc):
        return en.random_bernoulli(en.expit(loc))
        
    def random(self, loc):
        return numpy.random.randint(0, 2, loc.shape).astype(numpy.float32)


class ExponentialLayer(object):
    
    constraints = ['positive']

    def __init__(self):
        pass
        
    def prox(self, vis):
        return vis.clip(min=en.EPSILON)
        
    def mean(self, loc):
        return 1.0 / loc
        
    def inverse_mean(self, mean):
        return 1.0 / mean
        
    def log_partition_function(self, loc):
        return -numpy.log(loc)

    def sample_state(self, loc):
        return numpy.random.exponential(loc)
        
    def random(self, loc):
        return numpy.random.exponential(numpy.ones_like(loc))
        

# ---- FUNCTIONS ----- #
        
def get(key):
    if 'gauss' in key.lower():
        return GaussianLayer()
    elif 'ising' in key.lower():
        return IsingLayer()
    elif 'bern' in key.lower():
        return BernoulliLayer()
    elif 'expo' in key.lower():
        return ExponentialLayer()
    else:
        raise ValueError('Unknown layer type')
