import numpy
from scipy.stats import bernoulli
from scipy.special import expit


#----- LAYER CLASSES -----#

class Layer(object):

    def __init__(self, length):
        self.len = length
        
    def update_params(self, *args):
        pass
    
    def mean(self):
        pass
    
    def sample_state(self):
        pass


class GaussianLayer(Layer):
    
    def __init__(self, length):
        super().__init__(length)
        self.loc = numpy.zeros((self.len, 1), dtype=numpy.float32)
        self.scale = numpy.ones((self.len, 1), dtype=numpy.float32)
        
    def update_params(self, *args):
        self.loc[:] = args[0]
        self.scale[:] = args[1]
        
    def mean(self):
        return self.loc
    
    def sample_state(self):
        return self.loc + self.scale * numpy.random.normal(loc=0.0, scale=1.0, size=self.loc.shape)


class IsingLayer(Layer):

    def __init__(self, length):
        super().__init__(length)
        self.loc = numpy.zeros((self.len, 1), dtype=numpy.int8)

    def update_params(self, *args):
        self.loc[:] = args[0]
        
    def mean(self):
        return numpy.tanh(self.loc)

    def sample_state(self):
        return 2 * bernoulli.rvs(expit(self.loc)) - 1
        
        
class BernoulliLayer(Layer):
    
    def __init__(self, length):
        super().__init__(length)
        self.loc = numpy.zeros((self.len, 1), dtype=numpy.int8)
        
    def update_params(self, *args):
        self.loc[:] = args[0]
        
    def mean(self):
        return expit(self.loc)
        
    def sample_state(self):
        return bernoulli.rvs(expit(self.loc))


class ExponentialLayer(Layer):

    def __init__(self, length):
        super().__init__(length)
        self.loc = numpy.ones((self.len, 1), dtype=numpy.float32)
        
    def update_params(self, *args):
        self.loc[:] = args[0]
        
    def mean(self):
        return 1.0 / self.loc

    def sample_state(self):
        return numpy.random.exponential(self.loc)
        
# ---- FUNCTIONS ----- #
        
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
