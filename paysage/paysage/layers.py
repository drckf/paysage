import numpy, math
import numexpr as ne
from  numba import jit, vectorize

LOG2 = 0.6931471805599453

#----- LAYER CLASSES -----#

class GaussianLayer(object):
    
    def __init__(self):
        pass
        
    def prox(self, vis):
        return vis
        
    def mean(self, loc):
        return loc
        
    def log_partition_function(self, scale):
        return -numpy.log(scale)
    
    def sample_state(self, loc, scale):
        return loc + scale * numpy.random.normal(loc=0.0, scale=1.0, size=loc.shape)


class IsingLayer(object):

    def __init__(self):
        pass
        
    def prox(self, vis):
        return 2.0 * (vis > 0.0).astype(numpy.float32) - 1.0
        
    def mean(self, loc):
        return numpy.tanh(loc)
        
    def log_partition_function(self, loc):
        return -LOG2 + numpy.logaddexp(-loc, loc)
        
    def sample_state(self, loc):
        return random_ising_vector(expit(loc))
        
        
class BernoulliLayer(object):
    
    def __init__(self):
        pass
        
    def prox(self, vis):
        return (vis > 0.0).astype(numpy.float32)
        
    def mean(self, loc):
        return expit(loc)
        
    def log_partition_function(self, loc):
        return numpy.logaddexp(0, loc)
        
    def sample_state(self, loc):
        return random_bernoulli_vector(expit(loc))


class ExponentialLayer(object):

    def __init__(self):
        pass
        
    def prox(self, vis):
        return vis.clip(min=0.0)
        
    def mean(self, loc):
        return 1.0 / loc
        
    def log_partition_function(self, loc):
        return -numpy.log(loc)

    def sample_state(self, loc):
        return numpy.random.exponential(loc)
        

# ---- FUNCTIONS ----- #

@vectorize('float32(float32)', nopython=True)
def expit(x):
    result = (1.0 + numpy.tanh(x/2.0)) / 2.0
    return result
    
@jit('float32(float32)', nopython=True)
def random_bernoulli(p):
    r = numpy.random.rand()
    if p < r:
        return numpy.float32(0.0)
    else:
        return numpy.float32(1.0)
        
@jit('float32[:](float32[:])', nopython=True)
def random_bernoulli_vector(p):
    r = numpy.random.rand(len(p))
    result = numpy.zeros_like(p)
    for i in range(len(p)):
        if p[i] < r[i]:
            result[i] = numpy.float32(0.0)
        else:
            result[i] = numpy.float32(1.0)
    return result
 
@jit('float32[:](float32[:])', nopython=True)   
def random_ising_vector(p):
    result = numpy.float32(2.0) * random_bernoulli_vector(p) - numpy.float32(1.0)
    return result
        
def get(key):
    if 'gauss' in key.lower():
        return GaussianLayer
    elif 'ising' in key.lower():
        return IsingLayer
    elif 'bern' in key.lower():
        return BernoulliLayer
    elif 'expo' in key.lower():
        return ExponentialLayer
    else:
        raise ValueError('Unknown layer type')
