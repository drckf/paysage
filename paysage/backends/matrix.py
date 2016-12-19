import numpy, math
from numba import jit
import numexpr as ne

EPSILON = numpy.finfo(numpy.float32).eps

"""
Routines for matrix operations

"""


# ----- ELEMENTWISE ----- #

def elementwise_inverse(x):
    y = EPSILON + x
    return 1/y
    
def mix_inplace(w,x,y):
    ne.evaluate('w*x + (1-w)*y', out=x)
    
def square_mix_inplace(w,x,y):
    ne.evaluate('w*x + (1-w)*y*y', out=x)
    
def sqrt_div(x,y):
    z = EPSILON + y
    return ne.evaluate('x/sqrt(z)')
    
def normalize(x):
    y = EPSILON + x
    return x/numpy.sum(y)
    
@jit('float32(float32[:],float32[:])',nopython=True)
def squared_norm(a, b):
    result = numpy.float32(0.0)
    for i in range(len(a)):
        result += (a[i] - b[i])**2
    return result    
    
@jit('float32(float32[:],float32[:])',nopython=True)
def euclidean_distance(a, b):
    return math.sqrt(squared_norm(a, b))


# ----- THE FOLLOWING FUNCTIONS ARE THE MAIN BOTTLENECKS ----- #    
    
def batch_dot(vis, W, hid):
    return (numpy.dot(vis, W) * hid).sum(1).astype(numpy.float32)
    
def batch_outer(vis, hid):
    return numpy.dot(vis.T, hid) / len(vis)
    
def xM_plus_a(x,M,a,notrans=True):
    if notrans:
        return a + numpy.dot(x,M)
    else:
        return a + numpy.dot(x,M.T)
        
def xMy(x,M,y):
    return numpy.dot(x,numpy.dot(M,y))

# ------------------------------------------------------------ #    

# ----- SPECIALIZED MATRIX FUNCTIONS ----- #
    
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
    