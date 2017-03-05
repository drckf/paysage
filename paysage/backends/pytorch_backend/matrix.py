import numpy, torch

EPSILON = numpy.finfo(numpy.float32).eps

def to_numpy_array(tensor):
    """
    Return tensor as a numpy array.

    """
    return tensor.numpy()

def float_scalar(scalar):
    """
    Cast scalar to a 32-bit float.

    """
    return numpy.float32(scalar)

def float_tensor(tensor):
    """
    Cast tensor to a float tensor.

    """
    return torch.FloatTensor(tensor)

def shape(tensor):
    """
    Return a tuple with the shape of the tensor.

    """
    return tuple(tensor.size())

def ndim(tensor):
    """
    Return the number of dimensions of a tensor.

    """
    return tensor.ndimension()

def transpose(tensor):
    """
    Return the transpose of a tensor.

    """
    return tensor.t()

def zeros(shape):
    """
    Return a tensor of a specified shape filled with zeros.

    """
    return torch.zeros(shape)

def zeros_like(tensor):
    """
    Return a tensor of zeros with the same shape as the input tensor.

    """
    return zeros(shape(tensor))

def ones(shape):
    """
    Return a tensor of a specified shape filled with ones.

    """
    return torch.ones(shape)

def ones_like(tensor):
    """
    Return a tensor of ones with the same shape as the input tensor.

    """
    return ones(shape(tensor))

def diag(mat):
    """
    Return the diagonal elements of a matrix.

    """
    return mat.diag()

def diagonal_matrix(vec):
    """
    Return a matrix with vec along the diagonal.

    """
    return torch.diag(vec)

def identity(n):
    """
    Return the n-dimensional identity matrix.

    """
    return torch.eye(n)

def fill_diagonal(mat, val):
    """
    Fill the diagonal of the matirx with a specified value.
    In-place function.

    """
    for i in range(len(mat)):
        mat[i,i] = val

def sign(tensor):
    """
    Return the elementwise sign of a tensor.

    """
    return tensor.sign()

def clip(tensor, a_min=-numpy.inf, a_max=numpy.inf):
    """
    Return a tensor with its values clipped between a_min and a_max.

    """
    return tensor.clamp(a_min, a_max)

def clip_inplace(tensor, a_min=-numpy.inf, a_max=numpy.inf):
    """
    Clip the values of a tensor between a_min and a_max.
    In-place function.

    """
    return tensor.clamp_(a_min, a_max)

def tround(tensor):
    """
    Return a tensor with rounded elements.

    """
    return tensor.round()

def flatten(tensor):
    """
    Return a flattened tensor.

    """
    return tensor.view(int(numpy.prod(shape(tensor))))

def reshape(tensor, newshape):
    """
    Return tensor with a new shape.

    """
    return tensor.view(*newshape)

def dtype(tensor):
    """
    Return the type of the tensor.

    """
    raise tensor.type()


def mix_inplace(w,x,y):
    """
    Compute a weighted average of two matrices (x and y) and store the results in x.
    Useful for keeping track of running averages during training.

    x <- w * x + (1-w) * y

    """
    x.mul_(w)
    x.add_(y.mul(1-w))

def square_mix_inplace(w,x,y):
    """
    Compute a weighted average of two matrices (x and y^2) and store the results in x.
    Useful for keeping track of running averages of squared matrices during training.

    x < w x + (1-w) * y**2

    """
    x.mul_(w)
    x.add_(y.mul(y).mul(1-w))

def sqrt_div(x,y):
    """
    Elementwise division of x by sqrt(y).

    """
    return x.div(torch.sqrt(EPSILON + y))

def normalize(x):
    """
    Divide x by it's sum.

    """
    return x.div(torch.sum(EPSILON + x))


def norm(x):
    """
    Return the L2 norm of a tensor.

    """
    return x.norm()

def tmax(x, axis=None, keepdims=False):
    """
    Return the elementwise maximum of a tensor along the specified axis.

    """
    if axis is not None:
        tmp = x.max(dim=axis)[0]
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.max()

def tmin(x, axis=None, keepdims=False):
    """
    Return the elementwise minimum of a tensor along the specified axis.

    """
    if axis is not None:
        tmp = x.min(dim=axis)[0]
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.min()

def mean(x, axis=None, keepdims=False):
    """
    Return the mean of the elements of a tensor along the specified axis.

    """
    if axis is not None:
        tmp = x.mean(dim=axis)
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.mean()

def var(x, axis=None, keepdims=False):
    """
    Return the variance of the elements of a tensor along the specified axis.

    """
    if axis is not None:
        tmp = x.var(dim=axis)
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.var()

def std(x, axis=None, keepdims=False):
    """
    Return the standard deviation of the elements of a tensor along the specified axis.

    """
    if axis is not None:
        tmp = x.std(dim=axis)
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.std()

def tsum(x, axis=None, keepdims=False):
    """
    Return the sum of the elements of a tensor along the specified axis.

    """
    if axis is not None:
        tmp = x.sum(dim=axis)
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.sum()

def tprod(x, axis=None, keepdims=False):
    """
    Return the product of the elements of a tensor along the specified axis.

    """
    if axis is not None:
        tmp = x.prod(dim=axis)
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.prod()

def tany(x, axis=None, keepdims=False):
    """
    Return True if any elements of the input tensor are true along the
    specified axis.

    """
    tmp = tmax(x == True, axis=axis)
    if keepdims:
        return tmp
    else:
        return flatten(tmp)

def tall(x, axis=None, keepdims=False):
    """
    Return True if all elements of the input tensor are true along the
    specified axis.

    """
    tmp = tmin(x == True, axis=axis)
    if keepdims:
        return tmp
    else:
        return flatten(tmp)

def equal(x, y):
    """
    Elementwise if two tensors are equal.

    """
    return torch.eq(x, y)

def allclose(x, y):
    """
    Test if all elements in the two tensors are approximately equal.

    """
    return torch.max(torch.abs(x - y)) <= EPSILON

def not_equal(x, y):
    """
    Elementwise test if two tensors are not equal.

    """
    return torch.ne(x, y)

def greater(x, y):
    """
    Elementwise test if x > y.

    """
    return torch.gt(x, y)

def greater_equal(x, y):
    """
    Elementwise test if x >= y.

    """
    return torch.ge(x, y)

def lesser(x, y):
    """
    Elementwise test if x < y.

    """
    return torch.lt(x, y)

def lesser_equal(x, y):
    """
    Elementwise test if x <= y.

    """
    return torch.le(x, y)

def maximum(x, y):
    """
    Elementwise maximum of two tensors.

    """
    return torch.max(x, y)

def minimum(x, y):
    """
    Elementwise minimum of two tensors.

    """
    return torch.min(x, y)

def argmax(x, axis=None):
    """
    Compute the indices of the maximal elements in x along the specified axis.

    """
    if axis is not None:
        return x.max(dim=axis)[1]
    else:
        a,b = x.max(dim=0)
        index = a.max(dim=1)[1]
        return b[0, index[0,0]]

def argmin(x, axis=None):
    """
    Compute the indices of the minimal elements in x along the specified axis.

    """
    if axis is not None:
        return x.min(dim=axis)[1]
    else:
        a,b = x.min(dim=0)
        index = a.min(dim=1)[1]
        return b[0, index[0,0]]

def dot(a, b):
    """
    Compute the matrix/dot product of tensors a and b.

    """
    return a @ b

def outer(x,y):
    """
    Compute the outer product of vectors x and y.

    """
    return torch.ger(x, y)

def broadcast(vec, mat):
    """
    Like the numpy.broadcast_to function.

    """
    needs_transpose = (ndim(vec)==2 and not shape(vec)[0] == 1)
    flat = flatten(vec)
    result = flat.unsqueeze(0).expand(mat.size(0), flat.size(0))
    if needs_transpose:
        return transpose(result)
    else:
        return result

def affine(a,b,W):
    """
    Evaluate the affine transformation a + W b.

    """
    tmp = dot(W, b)
    tmp += broadcast(a, tmp)
    return tmp

def quadratic(a,b,W):
    """
    Evaluate the quadratic form a W b.

    """
    return a @ W @ b

def inv(mat):
    """
    Compute matrix inverse.

    """
    return mat.inverse()

def batch_dot(vis, W, hid, axis=1):
    """
    Let v by a L x N matrix where each row v_i is a visible vector.
    Let h be a L x M matrix where each row h_i is a hidden vector.
    And, let W be a N x M matrix of weights.
    Then, batch_dot(v,W,h) = \sum_i v_i^T W h_i
    Returns a vector.

    The actual computation is performed with a vectorized expression.

    """
    return tsum(dot(vis, W) * hid, axis)

def batch_outer(vis, hid):
    """
    Let v by a L x N matrix where each row v_i is a visible vector.
    Let h be a L x M matrix where each row h_i is a hidden vector.
    Then, batch_outer(v, h) = \sum_i v_i h_i^T
    Returns an N x M matrix.

    The actual computation is performed with a vectorized expression.

    """
    return dot(transpose(vis), hid)

def repeat(tensor, n, axis):
    """
    Repeat tensor n times along specified axis.

    """
    shapes  = tuple(n if i == axis else 1 for i in range(ndim(tensor)))
    return tensor.repeat(*shapes)

def stack(tensors, axis):
    """
    Stack tensors along the specified axis.

    """
    return torch.stack(tensors, dim=axis)

def hstack(tensors):
    """
    Concatenate tensors along the first axis.

    """
    return torch.stack(tensors, 1)

def vstack(tensors):
    """
    Concatenate tensors along the zeroth axis.

    """
    return torch.cat(tensors, 0)

def trange(start, end, step=1):
    """
    Generate a tensor like a python range.

    """
    return torch.range(start, end-1, step)

def euclidean_distance(a, b):
    """
    Compute the euclidean distance between two vectors.

    """
    return (a - b).norm()

def squared_euclidean_distance(a, b):
    """
    Compute the squared euclidean distance between two vectors.

    """
    return euclidean_distance(a, b)**2

def resample(tensor, n, replace=True):
    """
    Resample a tensor along the zeroth axis.

    """
    index = torch.LongTensor(
    numpy.random.choice(numpy.arange(len(tensor)), size=n, replace=replace)
    )
    return tensor.index_select(0, index)

def fast_energy_distance(minibatch, samples, downsample=100):
    """
    Compute an approximate energy distance between two tensors.

    """
    d1 = 0
    d2 = 0
    d3 = 0

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
