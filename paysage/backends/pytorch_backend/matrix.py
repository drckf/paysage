import numpy, torch
from multipledispatch import dispatch

EPSILON = numpy.finfo(numpy.float32).eps

# ----- TENSORS ----- #
"""
This section provides some wrappers to basic torch operations with arrays.

"""

def to_numpy_array(tensor):
    return tensor.numpy()

def float_scalar(scalar):
    return numpy.float32(scalar)

def float_tensor(tensor):
    return torch.FloatTensor(tensor)

def shape(tensor):
    return tuple(tensor.size())

def ndim(tensor):
    return tensor.ndimension()

def transpose(tensor):
    return torch.transpose(tensor)

def zeros(shape):
    return torch.zeros(shape)

def zeros_like(tensor):
    return zeros(shape(tensor))

def ones(shape):
    return torch.ones(shape)

def ones_like(tensor):
    return ones(shape(tensor))

def diag(mat):
    return torch.diag(mat)

def diagonal_matrix(vec):
    return torch.diag(vec)

def identity(n):
    return torch.eye(n)

def fill_diagonal(mat, val):
    for i in range(len(mat)):
        mat[i,i] = val

def sign(tensor):
    return torch.sign(tensor)

def clip(tensor, a_min=None, a_max=None):
    if a_min is None:
        return torch.clamp(tensor, max=a_max)
    elif a_max is None:
        return torch.clamp(tensor, min=a_min)
    else:
        return torch.clamp(tensor, min=a_min, max=a_max)

def clip_inplace(tensor, a_min=None, a_max=None):
    if a_min is None:
        return tensor.clamp_(tensor, max=a_max)
    elif a_max is None:
        return tensor.clamp_(tensor, min=a_min)
    else:
        return tensor.clamp_(tensor, min=a_min, max=a_max)

def tround(tensor):
    return torch.round(tensor)

def flatten(tensor):
    return tensor.view(int(numpy.prod(shape(tensor))))

def reshape(tensor, newshape):
    return tensor.view(*newshape)

def dtype(tensor):
    raise tensor.type()


######################

"""
Routines for matrix operations

"""

def mix_inplace(w,x,y):
    """
        Compute a weighted average of two matrices (x and y) and store the results in x.
        Useful for keeping track of running averages during training.

        x <- w * x + (1-w) * y

    """
    x *= w
    x += (1-w) * y

def square_mix_inplace(w,x,y):
    """
        Compute a weighted average of two matrices (x and y^2) and store the results in x.
        Useful for keeping track of running averages of squared matrices during training.

        x < w x + (1-w) * y**2

    """
    x *= w
    x += (1-w) * y * y

def sqrt_div(x,y):
    """
        Elementwise division of x by sqrt(y).

    """
    return x / torch.sqrt(EPSILON + y)

def normalize(x):
    """
        Divide x by it's sum.

    """
    return x / torch.sum(EPSILON + x)


# ----- THE FOLLOWING FUNCTIONS ARE THE MAIN BOTTLENECKS ----- #

def norm(x):
    return torch.norm(x)

def tmax(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.max(x, dim=axis)[0]
    else:
        return torch.max(x)

def tmin(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.min(x, dim=axis)[0]
    else:
        return torch.min(x)

def mean(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.mean(x, dim=axis)
    else:
        return torch.mean(x)

def var(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.var(x, dim=axis)
    else:
        return torch.var(x)

def std(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.std(x, dim=axis)
    else:
        return torch.std(x)

def tsum(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.sum(x, dim=axis)
    else:
        return torch.sum(x)

def tprod(x, axis=None, keepdims=False):
    if axis is not None:
        return torch.prod(x, dim=axis)
    else:
        return torch.prod(x)

def tany(x, axis=None, keepdims=False):
    return tmax(x == True, axis=axis)

def tall(x, axis=None, keepdims=False):
    return tmin(x == True, axis=axis)

def equal(x, y):
    return torch.eq(x, y)

def allclose(x, y):
    return torch.le(torch.abs(x - y), EPSILON)

def not_equal(x, y):
    return torch.neq(x, y)

def greater(x, y):
    return torch.gt(x, y)

def greater_equal(x, y):
    return torch.ge(x, y)

def lesser(x, y):
    return torch.lt(x, y)

def lesser_equal(x, y):
    return torch.le(x, y)

def maximum(x, y):
    return torch.max(x, y)

def minimum(x, y):
    return torch.min(x, y)

def argmax(x, axis=-1):
    if axis is not None:
        return torch.max(x, dim=axis)[1]
    else:
        a,b = torch.max(x, dim=0)
        index = torch.max(a, dim=1)[1]
        return b[0, index[0,0]]

def argmin(x, axis=-1):
    if axis is not None:
        return torch.min(x, dim=axis)[1]
    else:
        a,b = torch.min(x, dim=0)
        index = torch.min(a, dim=1)[1]
        return b[0, index[0,0]]

def dot(a,b):
    dims = ndim(a) * ndim(b)
    if dims == 4:
        return torch.mm(a, b)
    elif dims == 2:
        return torch.mv(a, b)
    elif dims == 1:
        return torch.dot(a, b)
    else:
        raise ValueError('Cannot determine appropriate matrix product')

def outer(x,y):
    return torch.ger(x, y)

def affine(a,b,W):
    raise NotImplementedError

def quadratic(a,b,W):
    raise NotImplementedError

def inv(mat):
    return torch.inverse(mat)

def batch_dot(vis, W, hid, axis=1):
    """
        Let v by a L x N matrix where each row v_i is a visible vector.
        Let h be a L x M matrix where each row h_i is a hidden vector.
        And, let W be a N x M matrix of weights.
        Then, batch_dot(v,W,h) = \sum_i v_i^T W h_i
        Returns a vector.

        The actual computation is performed with a vectorized expression.

    """
    raise NotImplementedError

def batch_outer(vis, hid):
    """
        Let v by a L x N matrix where each row v_i is a visible vector.
        Let h be a L x M matrix where each row h_i is a hidden vector.
        Then, batch_outer(v, h) = \sum_i v_i h_i^T
        Returns an N x M matrix.

        The actual computation is performed with a vectorized expression.

    """
    raise NotImplementedError

def repeat(tensor, n, axis):
    raise NotImplementedError

def stack(tensors, axis):
    raise NotImplementedError

def hstack(tensors):
    raise NotImplementedError

def vstack(tensors):
    raise NotImplementedError

def trange(start, end, step=1):
    raise NotImplementedError


# ------------------------------------------------------------ #

# ----- SPECIALIZED MATRIX FUNCTIONS ----- #

def squared_euclidean_distance(a, b):
    """
        Compute the squared euclidean distance between two vectors.

    """
    raise NotImplementedError

def euclidean_distance(a, b):
    """
        Compute the euclidean distance between two vectors.

    """
    raise NotImplementedError

def fast_energy_distance(minibatch, samples, downsample=100):
    raise NotImplementedError
