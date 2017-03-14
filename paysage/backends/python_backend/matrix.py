import numpy, math
from numba import jit
import numexpr as ne

def float_scalar(scalar):
    """
    Cast scalar to a 32-bit float.

    """
    return numpy.float32(scalar)

EPSILON = float_scalar(numpy.finfo(numpy.float32).eps)

def float_tensor(tensor):
    """
    Cast tensor to a float tensor.

    """
    return numpy.array(tensor, dtype=numpy.float32)

def to_numpy_array(tensor):
    """
    Return tensor as a numpy array.

    """
    return tensor

def shape(tensor):
    """
    Return a tuple with the shape of the tensor.

    """
    return tensor.shape

def ndim(tensor):
    """
    Return the number of dimensions of a tensor.

    """
    return tensor.ndim

def transpose(tensor):
    """
    Return the transpose of a tensor.

    """
    return numpy.transpose(tensor)

def zeros(shape):
    """
    Return a tensor of a specified shape filled with zeros.

    """
    return numpy.zeros(shape, dtype=numpy.float32)

def zeros_like(tensor):
    """
    Return a tensor of zeros with the same shape as the input tensor.

    """
    return zeros(shape(tensor))

def ones(shape):
    """
    Return a tensor of a specified shape filled with ones.

    """
    return numpy.ones(shape, dtype=numpy.float32)

def ones_like(tensor):
    """
    Return a tensor of ones with the same shape as the input tensor.

    """
    return ones(shape(tensor))

def diag(vec):
    """
    Return the diagonal elements of a matrix.

    """
    return numpy.diag(vec)

def diagonal_matrix(mat):
    """
    Return a matrix with vec along the diagonal.

    """
    return numpy.diag(mat)

def identity(n):
    """
    Return the n-dimensional identity matrix.

    """
    return numpy.identity(n, dtype=numpy.float32)

def fill_diagonal(mat, val):
    """
    Fill the diagonal of the matirx with a specified value.
    In-place function.

    """
    numpy.fill_diagonal(mat, val)

def sign(tensor):
    """
    Return the elementwise sign of a tensor.

    """
    return numpy.sign(tensor)

def clip(tensor, a_min=None, a_max=None):
    """
    Return a tensor with its values clipped between a_min and a_max.

    """
    return tensor.clip(a_min, a_max)

def clip_inplace(tensor, a_min=None, a_max=None):
    """
    Clip the values of a tensor between a_min and a_max.
    In-place function.

    """
    tensor.clip(a_min, a_max, out=tensor)

def tround(tensor):
    """
    Return a tensor with rounded elements.

    """
    return numpy.round(tensor)

def flatten(tensor):
    """
    Return a flattened tensor.

    """
    try:
        return tensor.ravel()
    except AttributeError:
        return tensor

def reshape(tensor, newshape):
    """
    Return tensor with a new shape.

    """
    return numpy.reshape(tensor, newshape)

def dtype(tensor):
    """
    Return the type of the tensor.

    """
    return tensor.dtype


def mix_inplace(w,x,y):
    """
    Compute a weighted average of two matrices (x and y) and store the results in x.
    Useful for keeping track of running averages during training.

    x <- w * x + (1-w) * y

    """
    ne.evaluate('w*x + (1-w)*y', out=x)

def square_mix_inplace(w,x,y):
    """
    Compute a weighted average of two matrices (x and y^2) and store the results in x.
    Useful for keeping track of running averages of squared matrices during training.

    x < w x + (1-w) * y**2

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

def norm(x):
    """
    Return the L2 norm of a tensor.

    """
    return numpy.linalg.norm(x)

def tmax(x, axis=None, keepdims=False):
    """
    Return the elementwise maximum of a tensor along the specified axis.

    """
    return numpy.max(x, axis=axis, keepdims=keepdims)

def tmin(x, axis=None, keepdims=False):
    """
    Return the elementwise minimum of a tensor along the specified axis.

    """
    return numpy.min(x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    """
    Return the mean of the elements of a tensor along the specified axis.

    """
    return numpy.mean(x, axis=axis, keepdims=keepdims)

def var(x, axis=None, keepdims=False):
    """
    Return the variance of the elements of a tensor along the specified axis.

    """
    return numpy.var(x, axis=axis, keepdims=keepdims, ddof=1)

def std(x, axis=None, keepdims=False):
    """
    Return the standard deviation of the elements of a tensor along the specified axis.

    """
    return numpy.std(x, axis=axis, keepdims=keepdims, ddof=1)

def tsum(x, axis=None, keepdims=False):
    """
    Return the sum of the elements of a tensor along the specified axis.

    """
    return numpy.sum(x, axis=axis, keepdims=keepdims)

def tprod(x, axis=None, keepdims=False):
    """
    Return the product of the elements of a tensor along the specified axis.

    """
    return numpy.prod(x, axis=axis, keepdims=keepdims)

def tany(x, axis=None, keepdims=False):
    """
    Return True if any elements of the input tensor are true along the
    specified axis.

    """
    return numpy.any(x, axis=axis, keepdims=keepdims)

def tall(x, axis=None, keepdims=False):
    """
    Return True if all elements of the input tensor are true along the
    specified axis.

    """
    return numpy.all(x, axis=axis, keepdims=keepdims)

def equal(x, y):
    """
    Elementwise if two tensors are equal.

    """
    return numpy.equal(x, y)

def allclose(x, y, rtol=1e-05, atol=1e-08):
    """
    Test if all elements in the two tensors are approximately equal.

    """
    return numpy.allclose(x, y, rtol=rtol, atol=atol)

def not_equal(x, y):
    """
    Elementwise test if two tensors are not equal.

    """
    return numpy.not_equal(x, y)

def greater(x, y):
    """
    Elementwise test if x > y.

    """
    return numpy.greater(x, y)

def greater_equal(x, y):
    """
    Elementwise test if x >= y.

    """
    return numpy.greater_equal(x, y)

def lesser(x, y):
    """
    Elementwise test if x < y.

    """
    return numpy.less(x, y)

def lesser_equal(x, y):
    """
    Elementwise test if x <= y.

    """
    return numpy.less_equal(x, y)

def maximum(x, y):
    """
    Elementwise maximum of two tensors.

    """
    return numpy.maximum(x, y)

def minimum(x, y):
    """
    Elementwise minimum of two tensors.

    """
    return numpy.minimum(x, y)

def argmax(x, axis):
    """
    Compute the indices of the maximal elements in x along the specified axis.

    """
    return numpy.argmax(x, axis=axis)

def argmin(x, axis):
    """
    Compute the indices of the minimal elements in x along the specified axis.

    """
    return numpy.argmin(x, axis=axis)

def dot(a,b):
    """
    Compute the matrix/dot product of tensors a and b.

    """
    return numpy.dot(a, b)

def outer(x,y):
    """
    Compute the outer product of vectors x and y.

    """
    return numpy.outer(x,y)


class BroadcastError(ValueError): pass

def broadcast(vec, matrix):
    """
    Broadcasts vec into the shape of matrix following numpy rules:

    vec ~ (N, 1) broadcasts to matrix ~ (N, M)
    vec ~ (1, N) and (N,) broadcast to matrix ~ (M, N)

    """
    try:
        return numpy.broadcast_to(vec, shape(matrix))
    except ValueError:
        raise BroadcastError('cannot broadcast vector of dimension {} \
onto matrix of dimension {}'.format(shape(vec), shape(matrix)))

def affine(a,b,W):
    """
    Evaluate the affine transformation a + W b.

    """
    return a + numpy.dot(W,b)

def quadratic(a,b,W):
    """
    Evaluate the quadratic form a W b.

    """
    return numpy.dot(a, numpy.dot(W, b))

def inv(mat):
    """
    Compute matrix inverse.

    """
    return numpy.linalg.inv(mat)

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

def repeat(tensor, n, axis):
    """
    Repeat tensor n times along specified axis.

    """
    return numpy.repeat(tensor, n, axis=axis)

def stack(tensors, axis):
    """
    Stack tensors along the specified axis.

    """
    return numpy.stack(tensors, axis=axis)

def hstack(tensors):
    """
    Concatenate tensors along the first axis.

    """
    return numpy.hstack(tensors)

def vstack(tensors):
    """
    Concatenate tensors along the zeroth axis.

    """
    return numpy.vstack(tensors)

def trange(start, end, step=1):
    """
    Generate a tensor like a python range.

    """
    return numpy.arange(start, end, step, dtype=numpy.float32)

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

@jit('float32[:,:](float32[:,:], int16, boolean)',nopython=True)
def resample(x, n, replace=True):
    """
    Resample a tensor along the zeroth axis.

    """
    index = numpy.random.choice(numpy.arange(len(x)),size=n,replace=replace)
    return x[index]

@jit('float32(float32[:,:],float32[:,:], int16)',nopython=True)
def fast_energy_distance(minibatch, samples, downsample=100):
    """
    Compute an approximate energy distance between two tensors.

    """
    d1 = numpy.float32(0)
    d2 = numpy.float32(0)
    d3 = numpy.float32(0)

    n = min(len(minibatch), downsample)
    m = min(len(samples), downsample)

    X = resample(minibatch, n, replace=True)
    Y = resample(samples, m, replace=True)

    for i in range(n):
        for j in range(i+1, n):
            d1 += euclidean_distance(X[i], X[j])
    d1 = 2.0 * d1 / (n*n - n)

    for i in range(m-1):
        for j in range(i+1, m):
            d2 += euclidean_distance(Y[i], Y[j])
    d2 = 2.0 * d2 / (m*m - m)

    for i in range(n):
        for j in range(m):
            d3 += euclidean_distance(X[i], Y[j])
    d3 = d3 / (n*m)

    return 2.0 * d3 - d2 - d1
