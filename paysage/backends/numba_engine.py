import numpy, math
from numba import jit, vectorize, void
import numexpr as ne

EPSILON = numpy.finfo(numpy.float32).eps

# ----- COMPILED FUNCTIONS ----- #

def running_mean(w, x, y):
    ne.evaluate('w*x + (1-w)*y', out=x)
    
def running_mean_square(w, x, y):
    ne.evaluate('w*x + (1-w)*y*y', out=x)
    
def sqrt_div(x,y):
    return ne.evaluate('x/sqrt(y)')

@vectorize('float32(float32)', nopython=True)
def expit(x):
    result = (1.0 + math.tanh(x/2.0)) / 2.0
    return result
    
@vectorize('float32(float32)', nopython=True)
def random_bernoulli(p):
    r = numpy.random.rand()
    if p < r:
        return numpy.float32(0.0)
    else:
        return numpy.float32(1.0)
 
@vectorize('float32(float32)', nopython=True)   
def random_ising(p):
    result = numpy.float32(2.0) * random_bernoulli(p) - numpy.float32(1.0)
    return result
    
# this function is not numba compiled
# using the regular, vectorized numpy expression is much faster
def batch_dot(vis, W, hid):
    return (numpy.dot(vis, W) * hid).sum(1).astype(numpy.float32)
    
def batch_outer(vis, hid):
    return numpy.dot(vis.T, hid) / len(vis)
    
@jit('float32[:](float32[:])',nopython=True)
def normalize(anarray):
    return anarray / numpy.sum(anarray)

@vectorize('float32(float32)',nopython=True)
def numba_exp(x):
    return math.exp(x)   
    
@jit('float32[:](float32[:],float32)',nopython=True)
def importance_weights(energies, temperature):
    gauge = energies - numpy.min(energies)
    return normalize(numba_exp(-gauge/temperature)) 
    
@jit('float32(float32[:],float32[:])',nopython=True)
def squared_norm(a, b):
    result = numpy.float32(0.0)
    for i in range(len(a)):
        result += (a[i] - b[i])**2
    return result    
    
@jit('float32(float32[:],float32[:])',nopython=True)
def euclidean_distance(a, b):
    return math.sqrt(squared_norm(a, b))
    
@jit('float32(float32[:,:],float32[:,:])',nopython=True)
def energy_distance(minibatch, samples):
    d1 = numpy.float32(0)
    d2 = numpy.float32(0)
    d3 = numpy.float32(0)
    
    n = len(minibatch)
    m = len(samples)

    for i in range(n-1):
        for j in range(i+1, n):
            d1 += euclidean_distance(minibatch[i], minibatch[j])
    d1 = 2.0 * d1 / (n*n - n)
    
    for i in range(m-1):
        for j in range(i+1, m):
            d2 += euclidean_distance(samples[i], samples[j])
    d2 = 2.0 * d2 / (m*m - m)
    
    for i in range(n):
        for j in range(m):
            d3 += euclidean_distance(minibatch[i], samples[j])
    d3 = d3 / (n*m)
    
    return 2.0 * d3 - d2 - d1
    
@jit('float32(float32[:,:],float32[:,:], int16)',nopython=True)
def fast_energy_distance(minibatch, samples, downsample=100):
    d1 = numpy.float32(0)
    d2 = numpy.float32(0)
    d3 = numpy.float32(0)
    
    n = min(len(minibatch), downsample)
    m = min(len(samples), downsample)
    
    index_1 = numpy.random.choice(numpy.arange(len(minibatch)), size=n)
    index_2 = numpy.random.choice(numpy.arange(len(samples)), size=m)

    for i in range(n-1):
        for j in range(i+1, n):
            d1 += euclidean_distance(minibatch[index_1[i]], minibatch[index_1[j]])
    d1 = 2.0 * d1 / (n*n - n)
    
    for i in range(m-1):
        for j in range(i+1, m):
            d2 += euclidean_distance(samples[index_1[i]], samples[index_2[j]])
    d2 = 2.0 * d2 / (m*m - m)
    
    for i in index_1:
        for j in index_2:
            d3 += euclidean_distance(minibatch[i], samples[j])
    d3 = d3 / (n*m)
    
    return 2.0 * d3 - d2 - d1
    