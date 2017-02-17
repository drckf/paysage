import numpy, math
from numba import jit
import numexpr as ne

EPSILON = numpy.finfo(numpy.float32).eps

# ----- TENSORS ----- #
"""
This section provides some wrappers to basic numpy operations with arrays.

"""

def flatten(tensor):
    return numpy.ravel(tensor)

def float_scalar(scalar):
    return numpy.float32(scalar)

def float_tensor(tensor):
    return numpy.array(tensor, dtype=numpy.float32)

def shape(tensor):
    return tensor.shape

def zeros(shape):
    return numpy.zeros(shape, dtype=numpy.float32)

def zeros_like(tensor):
    return zeros(shape(tensor))

def ones(shape):
    return numpy.ones(shape, dtype=numpy.float32)

def ones_like(tensor):
    return ones(shape(tensor))

def diag(vec):
    return numpy.diag(vec)

def diagonal_matrix(mat):
    return numpy.diag(mat)

def identity(n):
    return numpy.identity(n, dtype=numpy.float32)

def fill_diagonal(mat, val):
    return numpy.fill_diagonal(mat, val)

def sign(tensor):
    return numpy.sign(tensor)

def clip(tensor, a_min=None, a_max=None):
    return tensor.clip(a_min, a_max)

def clip_inplace(tensor, a_min=None, a_max=None):
    tensor.clip(a_min, a_max, out=tensor)

def round(tensor):
    return numpy.round(tensor)


######################

"""
Routines for matrix operations

"""

# ----- ELEMENTWISE ----- #

def elementwise_inverse(x):
    """
       Compute a safe element-wise inverse of a non-negative matrix.

    """
    y = EPSILON + x
    return 1/y

def mix_inplace(w,x,y):
    """
        Compute a weighted average of two matrices (x and y) and store the results in x.
        Useful for keeping track of running averages during training.

    """
    ne.evaluate('w*x + (1-w)*y', out=x)

def square_mix_inplace(w,x,y):
    """
        Compute a weighted average of two matrices (x and y^2) and store the results in x.
        Useful for keeping track of running averages of squared matrices during training.

    """
    ne.evaluate('w*x + (1-w)*y*y', out=x)

def sqrt_div(x,y):
    """
        Elementwise division of x by sqrt(y).

    """
    z = EPSILON + y
    return ne.evaluate('x/sqrt(z)')

def normalize(x):
    """
        Divide x by it's sum.

    """
    y = EPSILON + x
    return x/numpy.sum(y)


# ----- THE FOLLOWING FUNCTIONS ARE THE MAIN BOTTLENECKS ----- #

def norm(x):
    return numpy.linalg.norm(x)

def mean(x, axis=None, keepdims=False):
    return numpy.mean(x, axis=axis, keepdims=keepdims)

def var(x, axis=None, keepdims=False):
    return numpy.var(x, axis=axis, keepdims=keepdims)

def std(x, axis=None, keepdims=False):
    return numpy.std(x, axis=axis, keepdims=keepdims)

def tensor_sum(x, axis=None, keepdims=False):
    return numpy.sum(x, axis=axis, keepdims=keepdims)

def msum(x, axis=None, keepdims=False):
    return numpy.sum(x, axis=axis, keepdims=keepdims)

def dot(a,b):
    return numpy.dot(a, b)

def dot_plus(a,b,c):
    return numpy.dot(a,b) + c

def quadratic_form(x,M,y):
    return numpy.dot(x,numpy.dot(M,y))

def outer(x,y):
    return numpy.outer(x,y)

def batch_dot(vis, W, hid, axis=1):
    """
        Let v by a L x N matrix where each row v_i is a visible vector.
        Let h be a L x M matrix where each row h_i is a hidden vector.
        And, let W be a N x M matrix of weights.
        Then, batch_dot(v,W,h) = \sum_i v_i^T W h_i
        Returns a vector.

        The actual computation is performed with a vectorized expression.

    """
    return (numpy.dot(vis, W) * hid).sum(axis).astype(numpy.float32)

def batch_outer(vis, hid):
    """
        Let v by a L x N matrix where each row v_i is a visible vector.
        Let h be a L x M matrix where each row h_i is a hidden vector.
        Then, batch_outer(v, h) = \sum_i v_i h_i^T
        Returns an N x M matrix.

        The actual computation is performed with a vectorized expression.

    """
    return numpy.dot(vis.T, hid)

def xM_plus_a(x,M,a,trans=False):
    if not trans:
        return a + numpy.dot(x,M)
    else:
        return a + numpy.dot(x,M.T)

def xMy(x,M,y):
    return numpy.dot(x,numpy.dot(M,y))

def inv(mat):
    return numpy.linalg.inv(mat)

# ------------------------------------------------------------ #

# ----- SPECIALIZED MATRIX FUNCTIONS ----- #

@jit('float32(float32[:],float32[:])',nopython=True)
def squared_euclidean_distance(a, b):
    """
        Compute the squared euclidean distance between two vectors.

    """
    result = numpy.float32(0.0)
    for i in range(len(a)):
        result += (a[i] - b[i])**2
    return result

@jit('float32(float32[:],float32[:])',nopython=True)
def euclidean_distance(a, b):
    """
        Compute the euclidean distance between two vectors.

    """
    return math.sqrt(squared_euclidean_distance(a, b))

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
