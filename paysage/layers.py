import numpy
from . import backends as B

#----- LAYER CLASSES -----#

class GaussianLayer(object):
        
    def __init__(self):
        pass
        
    def prox(self, vis):
        return vis
        
    def mean(self, loc):
        return loc
        
    def inverse_mean(self, mean):
        return mean
        
    def log_partition_function(self, scale):
        return -B.log(scale)
    
    def sample_state(self, loc, scale):
        return loc + scale * numpy.random.normal(loc=0.0, scale=1.0, size=loc.shape).astype(numpy.float32)
        
    def random(self, loc, scale):
        return numpy.random.normal(loc=0.0, scale=1.0, size=loc.shape).astype(numpy.float32)


class IsingLayer(object):
    
    def __init__(self):
        pass
        
    def prox(self, vis):
        return 2.0 * (vis > 0.0).astype(numpy.float32) - 1.0
        
    def mean(self, loc):
        return B.tanh(loc)
        
    def inverse_mean(self, mean):
        return B.atanh(mean)
        
    def log_partition_function(self, loc):
        return B.logcosh(loc)
        
    def sample_state(self, loc):
        return B.random_ising(B.expit(loc))
        
    def random(self, loc):
        return 2 * numpy.random.randint(0, 2, loc.shape).astype(numpy.float32) - 1
        
        
class BernoulliLayer(object):
        
    def __init__(self):
        pass
        
    def prox(self, vis):
        return (vis > 0.0).astype(numpy.float32)
        
    def mean(self, loc):
        return B.expit(loc)
        
    def inverse_mean(self, mean):
        return B.logit(mean)
        
    def log_partition_function(self, loc):
        return B.softplus(loc)
        
    def sample_state(self, loc):
        return B.random_bernoulli(B.expit(loc))
        
    def random(self, loc):
        return numpy.random.randint(0, 2, loc.shape).astype(numpy.float32)


class ExponentialLayer(object):
    
    def __init__(self):
        pass
        
    def prox(self, vis):
        return vis.clip(min=B.EPSILON)
        
    def mean(self, loc):
        return B.elementwise_inverse(loc)
        
    def inverse_mean(self, mean):
        return B.elementwise_inverse(mean)
        
    def log_partition_function(self, loc):
        return -B.log(loc)

    def sample_state(self, loc):
        return numpy.random.exponential(loc).astype(numpy.float32)
        
    def random(self, loc):
        return numpy.random.exponential(numpy.ones_like(loc)).astype(numpy.float32)
        

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
