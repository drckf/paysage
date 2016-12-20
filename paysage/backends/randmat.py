import numpy
from numba import jit, vectorize

# ----- FUNCTIONS ----- #
    

def random_bernoulli(p):
    r = numpy.random.rand(*p.shape)
    return numpy.float32(r<p)
 
def random_ising(p):
    result = 2*random_bernoulli(p)-1
    return result
    