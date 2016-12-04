import numpy
from numba import jit, vectorize

# ----- COMPILED FUNCTIONS ----- #

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
    
@jit('float32[:](float32[:,:],float32[:,:],float32[:,:])',nopython=True)
def batch_dot(vis, W, hid):
    result = numpy.zeros(len(vis), dtype=numpy.float32)
    for i in range(len(vis)):
        result[i] = numpy.dot(vis[i], numpy.dot(W, hid[i]))
    return result

@jit('float32[:,:](float32[:],float32[:])',nopython=True)
def outer(vis, hid):
    result = numpy.zeros((len(vis), len(hid)), dtype=numpy.float32)
    for i in range(len(vis)):
        for u in range(len(hid)):
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
    result = numpy.zeros((vis.shape[1], hid.shape[1]), dtype=numpy.float32)
    for i in range(len(vis)):
        outer_inplace(vis[i], hid[i], result)
    return result / len(vis)
    
@jit('float32[:](float32[:])',nopython=True)
def normalize(anarray):
    return anarray / numpy.sum(numpy.abs(anarray))
    
@jit('float32[:](float32[:],float32)',nopython=True)
def importance_weights(energies, temperature):
    gauge = energies - numpy.min(energies)
    return normalize(numpy.exp(-gauge/temperature)) 
    
@jit('float32(float32[:,:],float32[:,:])',nopython=True)
def energy_distance(minibatch, samples):
    d1 = numpy.float32(0)
    d2 = numpy.float32(0)
    d3 = numpy.float32(0)

    for i in range(len(minibatch)):
        for j in range(len(minibatch)):
            d1 += numpy.linalg.norm(minibatch[i] - minibatch[j])
    d1 = d1 / (len(minibatch)**2 - len(minibatch))
    
    for i in range(len(samples)):
        for j in range(len(samples)):
            d2 += numpy.linalg.norm(samples[i] - samples[j])
    d2 = d2 / (len(samples)**2 - len(samples))
    
    for i in range(len(minibatch)):
        for j in range(len(samples)):
            d3 += numpy.linalg.norm(minibatch[i] - samples[j])
    d3 = d3 / (len(minibatch)*len(samples))
    
    return 2*d3 - d2 - d1