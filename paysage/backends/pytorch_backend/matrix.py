import numpy, torch
from . import typedef as T

def add_dicts_inplace(dict1, dict2) -> None:
    """
    Entrywise addition of dict2 to dict1.
    Modifies dict1 in place.

    """

    for key in dict2:
        dict1[key] += dict2[key]

def subtract_dicts_inplace(dict1, dict2) -> None:
    """
    Entrywise subtraction of dict2 from dict1.
    Modifies dict1 in place.

    """

    for key in dict2:
        dict1[key] -= dict2[key]

def multiply_dict_inplace(dict1, scalar: T.Scalar) -> None:
    """
    Entrywise multiplication of dict1 by scalar.
    Modifies dict1 in place.

    """

    for key in dict1:
        dict1[key] *= scalar

def float_scalar(scalar: T.Scalar) -> float:
    """
    Cast scalar to a float.

    """
    return float(scalar)

EPSILON = float_scalar(numpy.finfo(numpy.float32).eps)

def float_tensor(tensor: T.Tensor) -> T.FloatTensor:
    """
    Cast tensor to a float tensor.

    """
    try:
        # tensor is a numpy object
        return torch.FloatTensor(tensor.astype(float))
    except Exception:
        # tensor is a torch object
        return tensor.float()

def to_numpy_array(tensor: T.Tensor) -> T.NumpyTensor:
    """
    Return tensor as a numpy array.

    """
    try:
        return tensor.numpy()
    except Exception:
        return numpy.array(tensor)

def shape(tensor: T.TorchTensor) -> T.Tuple[int]:
    """
    Return a tuple with the shape of the tensor.

    """
    return tuple(tensor.size())

def ndim(tensor: T.TorchTensor) -> int:
    """
    Return the number of dimensions of a tensor.

    """
    return tensor.ndimension()

def transpose(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return the transpose of a tensor.

    """
    return tensor.t()

def zeros(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Return a tensor of a specified shape filled with zeros.

    """
    return torch.zeros(shape)

def zeros_like(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return a tensor of zeros with the same shape as the input tensor.

    """
    return zeros(shape(tensor))

def ones(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Return a tensor of a specified shape filled with ones.

    """
    return torch.ones(shape)

def ones_like(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return a tensor of ones with the same shape as the input tensor.

    """
    return ones(shape(tensor))

def diag(mat: T.TorchTensor) -> T.TorchTensor:
    """
    Return the diagonal elements of a matrix.

    """
    return mat.diag()

def diagonal_matrix(vec: T.TorchTensor) -> T.TorchTensor:
    """
    Return a matrix with vec along the diagonal.

    """
    return torch.diag(vec)

def identity(n: int) -> T.FloatTensor:
    """
    Return the n-dimensional identity matrix.

    """
    return torch.eye(n)

def fill_diagonal(mat: T.FloatTensor, val: T.Scalar) -> None:
    """
    Fill the diagonal of the matirx with a specified value.
    In-place function.

    """
    for i in range(len(mat)):
        mat[i,i] = val

def sign(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return the elementwise sign of a tensor.

    """
    return tensor.sign()

def clip(tensor: T.FloatTensor,
         a_min: T.Scalar = -numpy.inf,
         a_max: T.Scalar = numpy.inf) -> T.FloatTensor:
    """
    Return a tensor with its values clipped between a_min and a_max.

    """
    return tensor.clamp(a_min, a_max)

def clip_inplace(tensor: T.FloatTensor,
                 a_min: T.Scalar = -numpy.inf,
                 a_max: T.Scalar = numpy.inf) -> None:
    """
    Clip the values of a tensor between a_min and a_max.
    In-place function.

    """
    return torch.clamp(tensor, a_min, a_max, out=tensor)

def tround(tensor: T.FloatTensor) -> T.FloatTensor:
    """
    Return a tensor with rounded elements.

    """
    return tensor.round()

def flatten(tensor: T.FloatingPoint) -> T.FloatingPoint:
    """
    Return a flattened tensor.

    """
    try:
        return tensor.view(int(numpy.prod(shape(tensor))))
    except AttributeError:
        return tensor

def reshape(tensor: T.FloatTensor,
            newshape: T.Tuple[int]) -> T.FloatTensor:
    """
    Return tensor with a new shape.

    """
    return tensor.view(*newshape)

def dtype(tensor: T.FloatTensor) -> type:
    """
    Return the type of the tensor.

    """
    return tensor.type()

def mix_inplace(w: T.Scalar,
                x: T.FloatTensor,
                y: T.FloatTensor) -> None:
    """
    Compute a weighted average of two matrices (x and y) and store the results in x.
    Useful for keeping track of running averages during training.

    x <- w * x + (1-w) * y

    """
    x.mul_(w)
    x.add_(y.mul(1-w))

def square_mix_inplace(w: T.Scalar,
                       x: T.FloatTensor,
                       y: T.FloatTensor) -> None:
    """
    Compute a weighted average of two matrices (x and y^2) and store the results in x.
    Useful for keeping track of running averages of squared matrices during training.

    x < w x + (1-w) * y**2

    """
    x.mul_(w)
    x.add_(y.mul(y).mul(1-w))

def sqrt_div(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise division of x by sqrt(y).

    """
    return x.div(torch.sqrt(EPSILON + y))

def normalize(x: T.FloatTensor) -> T.FloatTensor:
    """
    Divide x by it's sum.

    """
    return x.div(torch.sum(EPSILON + x))

def norm(x: T.FloatTensor) -> float:
    """
    Return the L2 norm of a tensor.

    """
    return x.norm()

def tmax(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
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

def tmin(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
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

def mean(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
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

def var(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
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

def std(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
    """
    Return the standard deviation of the elements of a tensor along the
    specified axis.

    """
    if axis is not None:
        tmp = x.std(dim=axis)
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.std()

def tsum(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
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

def tprod(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
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

def equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test for if two tensors are equal.

    """
    return torch.eq(x, y)

def not_equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if two tensors are not equal.

    """
    return torch.ne(x, y)

def greater(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x > y.

    """
    return torch.gt(x, y)

def greater_equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x >= y.

    """
    return torch.ge(x, y)

def lesser(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x < y.

    """
    return torch.lt(x, y)

def lesser_equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x <= y.

    """
    return torch.le(x, y)

def tany(x: T.Tensor,
         axis: int = None,
         keepdims: bool = False) -> T.Boolean:
    """
    Return True if any elements of the input tensor are true along the
    specified axis.

    """
    tmp = tmax(x.ne(0), axis=axis)
    if keepdims:
        return tmp
    else:
        return flatten(tmp)

def tall(x: T.Tensor,
         axis: int = None,
         keepdims: bool = False) -> T.Boolean:
    """
    Return True if all elements of the input tensor are true along the
    specified axis.

    """
    tmp = tmin(x.ne(0), axis=axis)
    if keepdims:
        return tmp
    else:
        return flatten(tmp)

def allclose(x: T.FloatTensor,
             y: T.FloatTensor,
             rtol: float = 1e-05,
             atol: float = 1e-08) -> bool:
    """
    Test if all elements in the two tensors are approximately equal.

    """
    return tall(torch.abs(x - y).le((atol + rtol * torch.abs(y))))

def maximum(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise maximum of two tensors.

    """
    return torch.max(x, y)

def minimum(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise minimum of two tensors.

    """
    return torch.min(x, y)

def argmax(x: T.FloatTensor, axis: int) -> T.LongTensor:
    """
    Compute the indices of the maximal elements in x along the specified axis.

    """
    # needs flatten because numpy argmax always returns a 1-D array
    return flatten(x.max(dim=axis)[1])

def argmin(x: T.FloatTensor, axis: int = None) -> T.LongTensor:
    """
    Compute the indices of the minimal elements in x along the specified axis.

    """
    # needs flatten because numpy argmin always returns a 1-D array
    return flatten(x.min(dim=axis)[1])

def dot(a: T.FloatTensor, b: T.FloatTensor) -> T.FloatingPoint:
    """
    Compute the matrix/dot product of tensors a and b.

    """
    return a @ b

def outer(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Compute the outer product of vectors x and y.

    """
    return torch.ger(x, y)

class BroadcastError(ValueError): pass

def broadcast(vec: T.FloatTensor, matrix: T.FloatTensor) -> T.FloatTensor:
    """
    Broadcasts vec into the shape of matrix following numpy rules:

    vec ~ (N, 1) broadcasts to matrix ~ (N, M)
    vec ~ (1, N) and (N,) broadcast to matrix ~ (M, N)

    """
    try:
        if ndim(vec) == 1:
            if ndim(matrix) == 1:
                return vec
            return vec.unsqueeze(0).expand(matrix.size(0), matrix.size(1))
        else:
            return vec.expand(matrix.size(0), matrix.size(1))
    except ValueError:
        raise BroadcastError('cannot broadcast vector of dimension {} \
onto matrix of dimension {}'.format(shape(vec), shape(matrix)))

def affine(a: T.FloatTensor,
           b: T.FloatTensor,
           W: T.FloatTensor) -> T.FloatTensor:
    """
    Evaluate the affine transformation a + W b.

    """
    tmp = dot(W, b)
    if ndim(tmp) > ndim(a):
        tmp += broadcast(a, tmp)
    else:
        tmp += a
    return tmp

def quadratic(a: T.FloatTensor,
              b: T.FloatTensor,
              W: T.FloatTensor) -> T.FloatTensor:
    """
    Evaluate the quadratic form a W b.

    """
    return a @ W @ b

def inv(mat: T.FloatTensor) -> T.FloatTensor:
    """
    Compute matrix inverse.

    """
    return mat.inverse()

def batch_dot(vis: T.FloatTensor,
              W: T.FloatTensor,
              hid: T.FloatTensor,
              axis: int = 1) -> T.FloatTensor:
    """
    Let v by a L x N matrix where each row v_i is a visible vector.
    Let h be a L x M matrix where each row h_i is a hidden vector.
    And, let W be a N x M matrix of weights.
    Then, batch_dot(v,W,h) = \sum_i v_i^T W h_i
    Returns a vector.

    The actual computation is performed with a vectorized expression.

    """
    return tsum(dot(vis, W) * hid, axis)

def batch_outer(vis: T.FloatTensor, hid: T.FloatTensor) -> T.FloatTensor:
    """
    Let v by a L x N matrix where each row v_i is a visible vector.
    Let h be a L x M matrix where each row h_i is a hidden vector.
    Then, batch_outer(v, h) = \sum_i v_i h_i^T
    Returns an N x M matrix.

    The actual computation is performed with a vectorized expression.

    """
    return dot(transpose(vis), hid)

def repeat(tensor: T.FloatTensor, n: int) -> T.FloatTensor:
    """
    Repeat tensor n times along specified axis.

    """
    # current implementation only works for vectors
    assert ndim(tensor) == 1
    return flatten(tensor.unsqueeze(1).repeat(1, n))

def stack(tensors: T.Iterable[T.FloatTensor], axis: int) -> T.FloatTensor:
    """
    Stack tensors along the specified axis.

    """
    return torch.stack(tensors, dim=axis)

def hstack(tensors: T.Iterable[T.FloatTensor]) -> T.FloatTensor:
    """
    Concatenate tensors along the first axis.

    """
    if ndim(tensors[0]) == 1:
        return torch.cat(tensors, 0)
    else:
        return torch.cat(tensors, 1)

def vstack(tensors: T.Iterable[T.FloatTensor]) -> T.FloatTensor:
    """
    Concatenate tensors along the zeroth axis.

    """
    if ndim(tensors[0]) == 1:
        return torch.stack(tensors, 0)
    else:
        return torch.cat(tensors, 0)

def trange(start: int, end: int, step: int = 1) -> T.FloatTensor:
    """
    Generate a tensor like a python range.

    """
    return torch.range(start, end-1, step)

def euclidean_distance(a: T.FloatTensor, b: T.FloatTensor) -> float:
    """
    Compute the euclidean distance between two vectors.

    """
    return (a - b).norm()

def squared_euclidean_distance(a: T.FloatTensor,
                               b: T.FloatTensor) -> float:
    """
    Compute the squared euclidean distance between two vectors.

    """
    return euclidean_distance(a, b)**2

def resample(tensor: T.FloatTensor,
             n: int,
             replace: bool = True) -> T.FloatTensor:
    """
    Resample a tensor along the zeroth axis.

    """
    index = torch.LongTensor(
    numpy.random.choice(numpy.arange(len(tensor)), size=n, replace=replace)
    )
    return tensor.index_select(0, index)

def fast_energy_distance(minibatch: T.FloatTensor,
                         samples: T.FloatTensor,
                         downsample: int = 100) -> float:
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
