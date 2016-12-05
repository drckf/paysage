import numpy, math
from numba import jit, vectorize

EPSILON = numpy.finfo(numpy.float32).eps

# ----- COMPILED FUNCTIONS ----- #

@vectorize('float32(float32)', nopython=True)
def expit(x):
    result = (1.0 + math.tanh(x/2.0)) / 2.0
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
    n = len(p)
    r = numpy.random.rand(n)
    result = numpy.zeros_like(p)
    for i in range(n):
        if p[i] < r[i]:
            result[i] = numpy.float32(0.0)
        else:
            result[i] = numpy.float32(1.0)
    return result
 
@jit('float32[:](float32[:])', nopython=True)   
def random_ising_vector(p):
    result = numpy.float32(2.0) * random_bernoulli_vector(p) - numpy.float32(1.0)
    return result
    
@jit('float32[:](float32[:,:],float32[:,:],float32[:,:])',nopython=True)
def batch_dot(vis, W, hid):
    n = len(vis)
    result = numpy.zeros(n, dtype=numpy.float32)
    for i in range(n):
        result[i] = numpy.dot(vis[i], numpy.dot(W, hid[i]))
    return result

@jit('float32[:,:](float32[:],float32[:])',nopython=True)
def outer(vis, hid):
    n = len(vis)
    m = len(hid)
    result = numpy.zeros((n, m), dtype=numpy.float32)
    for i in range(n):
        for u in range(m):
            result[i][u] = vis[i] * hid[u]
    return result
    
@jit('float32[:,:](float32[:],float32[:], float32[:,:])',nopython=True)
def outer_inplace(vis, hid, result):
    for i in range(len(vis)):
        for u in range(len(hid)):
            result[i][u] += vis[i] * hid[u]
    return result
    
@jit('float32[:,:](float32[:,:],float32[:,:])',nopython=True)
def batch_outer(vis, hid):
    n = len(vis)
    result = numpy.zeros((vis.shape[1], hid.shape[1]), dtype=numpy.float32)
    for i in range(n):
        outer_inplace(vis[i], hid[i], result)
    return result / n
    
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
def euclidean_distance(a, b):
    result = numpy.float32(0.0)
    for i in range(len(a)):
        result += (a[i] - b[i])**2
    return math.sqrt(result)
    
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