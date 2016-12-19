import numpy

# ----- FUNCTIONS ----- #
    
def random_bernoulli(p):
    r = numpy.random.rand(*p.shape)
    return numpy.float32(p < r)
 
def random_ising(p):
    result = 2*random_bernoulli(p)-1
    return result
    