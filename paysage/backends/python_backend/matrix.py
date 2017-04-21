import numpy, math
from numba import jit
import numexpr as ne
from . import typedef as T

def float_scalar(scalar: T.Scalar) -> float:
    """
    Cast scalar to a 32-bit float.

    Args:
        scalar: A scalar quantity:

    Returns:
        numpy.float32: Scalar converted to floating point.

    """
    return numpy.float32(scalar)

EPSILON = float_scalar(numpy.finfo(numpy.float32).eps)

def float_tensor(tensor: T.Tensor) -> T.Tensor:
    """
    Cast tensor to a float tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: Tensor converted to floating point.

    """
    return numpy.array(tensor, dtype=numpy.float32)

def to_numpy_array(tensor: T.Tensor) -> T.Tensor:
    """
    Return tensor as a numpy array.

    Args:
        tensor: A tensor.

    Returns:
        tensor: Tensor converted to a numpy array.

    """
    return tensor

def shape(tensor: T.Tensor) -> T.Tuple[int]:
    """
    Return a tuple with the shape of the tensor.

    Args:
        tensor: A tensor:

    Returns:
        tuple: A tuple of integers describing the shape of the tensor.

    """
    return tensor.shape

def ndim(tensor: T.Tensor) -> int:
    """
    Return the number of dimensions of a tensor.

    Args:
        tensor: A tensor:

    Returns:
        int: The number of dimensions of the tensor.

    """
    return tensor.ndim

def num_elements(tensor: T.Tensor) -> int:
    """
    Return the number of elements in a tensor.

    Args:
        tensor: A tensor:

    Returns:
        int: The number of elements in the tensor.

    """
    return numpy.prod(shape(tensor))

def transpose(tensor: T.Tensor) -> T.Tensor:
    """
    Return the transpose of a tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: The transpose (exchange of rows and columns) of the tensor.

    """
    return numpy.transpose(tensor)

def zeros(shape: T.Tuple[int]) -> T.Tensor:
    """
    Return a tensor of a specified shape filled with zeros.

    Args:
        shape: The shape of the desired tensor.

    Returns:
        tensor: A tensor of zeros with the desired shape.

    """
    return numpy.zeros(shape, dtype=numpy.float32)

def zeros_like(tensor: T.Tensor) -> T.Tensor:
    """
    Return a tensor of zeros with the same shape as the input tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor of zeros with the same shape.

    """
    return zeros(shape(tensor))

def ones(shape: T.Tuple[int]) -> T.Tensor:
    """
    Return a tensor of a specified shape filled with ones.

    Args:
        shape: The shape of the desired tensor.

    Returns:
        tensor: A tensor of ones with the desired shape.

    """
    return numpy.ones(shape, dtype=numpy.float32)

def ones_like(tensor: T.Tensor) -> T.Tensor:
    """
    Return a tensor of ones with the same shape as the input tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor with the same shape.

    """
    return ones(shape(tensor))

def diagonal_matrix(vec: T.Tensor) -> T.Tensor:
    """
    Return a matrix with vec along the diagonal.

    Args:
        vec: A vector (i.e., 1D tensor).

    Returns:
        tensor: A matrix with the elements of vec along the diagonal,
                and zeros elsewhere.

    """
    return numpy.diag(vec)

def diag(mat: T.Tensor) -> T.Tensor:
    """
    Return the diagonal elements of a matrix.

    Args:
        mat: A tensor.

    Returns:
        tensor: A vector (i.e., 1D tensor) containing the diagonal
                elements of mat.

    """
    return numpy.diag(mat)

def identity(n: int) -> T.Tensor:
    """
    Return the n-dimensional identity matrix.

    Args:
        n: The desired size of the tensor.

    Returns:
        tensor: The n x n identity matrix with ones along the diagonal
                and zeros elsewhere.

    """
    return numpy.identity(n, dtype=numpy.float32)

def fill_diagonal(mat: T.Tensor, val: T.Scalar) -> T.Tensor:
    """
    Fill the diagonal of the matirx with a specified value.

    Note:
        Modifies mat in place.

    Args:
        mat: A tensor.
        val: The value to put along the diagonal.

    Returns:
        None

    """
    numpy.fill_diagonal(mat, val)

def sign(tensor: T.Tensor) -> T.Tensor:
    """
    Return the elementwise sign of a tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: The sign of the elements in the tensor.

    """
    return numpy.sign(tensor)

def clip(tensor: T.Tensor, a_min: T.Scalar=None,
         a_max: T.Scalar=None) -> T.Tensor:
    """
    Return a tensor with its values clipped between a_min and a_max.

    Args:
        tensor: A tensor.
        a_min (optional): The desired lower bound on the elements of the tensor.
        a_max (optional): The desired upper bound on the elements of the tensor.

    Returns:
        tensor: A new tensor with its values clipped between a_min and a_max.

    """
    return tensor.clip(a_min, a_max)

def clip_inplace(tensor: T.Tensor, a_min: T.Scalar=None,
                 a_max: T.Scalar=None) -> None:
    """
    Clip the values of a tensor between a_min and a_max.

    Note:
        Modifies tensor in place.

    Args:
        tensor: A tensor.
        a_min (optional): The desired lower bound on the elements of the tensor.
        a_max (optional): The desired upper bound on the elements of the tensor.

    Returns:
        None

    """
    tensor.clip(a_min, a_max, out=tensor)

def tround(tensor: T.Tensor) -> T.Tensor:
    """
    Return a tensor with rounded elements.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor rounded to the nearest integer (still floating point).

    """
    return numpy.round(tensor)

def flatten(tensor: T.FloatingPoint) -> T.FloatingPoint:
    """
    Return a flattened tensor.

    Args:
        tensor: A tensor or scalar.

    Returns:
        result: If arg is a tensor, return a flattened 1D tensor.
                If arg is a scalar, return the scalar.

    """
    try:
        return tensor.ravel()
    except AttributeError:
        return tensor

def reshape(tensor: T.Tensor, newshape: T.Tuple[int]) -> T.Tensor:
    """
    Return tensor with a new shape.

    Args:
        tensor: A tensor.
        newshape: The desired shape.

    Returns:
        tensor: A tensor with the desired shape.

    """
    return numpy.reshape(tensor, newshape)

def unsqueeze(tensor: T.Tensor, axis: int) -> T.Tensor:
    """
    Return tensor with a new axis inserted.

    Args:
        tensor: A tensor.
        axis: The desired axis.

    Returns:
        tensor: A tensor with the new axis inserted.

    """
    return numpy.expand_dims(tensor, axis)

def dtype(tensor: T.Tensor) -> type:
    """
    Return the type of the tensor.

    Args:
        tensor: A tensor.

    Returns:
        type: The type of the elements in the tensor.

    """
    return tensor.dtype

def mix_inplace(w: T.Scalar, x: T.Tensor, y: T.Tensor) -> None:
    """
    Compute a weighted average of two matrices (x and y) and store the results in x.
    Useful for keeping track of running averages during training.

    x <- w * x + (1-w) * y

    Note:
        Modifies x in place.

    Args:
        w: The mixing coefficient between 0 and 1 .
        x: A tensor.
        y: A tensor:

    Returns:
        None

    """
    ne.evaluate('w*x + (1-w)*y', out=x)

def square_mix_inplace(w: T.Scalar, x: T.Tensor, y: T.Tensor) -> None:
    """
    Compute a weighted average of two matrices (x and y^2) and store the results in x.
    Useful for keeping track of running averages of squared matrices during training.

    x <- w x + (1-w) * y**2

    Note:
        Modifies x in place.

    Args:
        w: The mixing coefficient between 0 and 1 .
        x: A tensor.
        y: A tensor:

    Returns:
        None

    """
    ne.evaluate('w*x + (1-w)*y*y', out=x)

def sqrt_div(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Elementwise division of x by sqrt(y).

    Args:
        x: A tensor:
        y: A non-negative tensor.

    Returns:
        tensor: Elementwise division of x by sqrt(y).

    """
    z = EPSILON + y
    return ne.evaluate('x/sqrt(z)')

def normalize(x: T.Tensor) -> T.Tensor:
    """
    Divide x by it's sum.

    Args:
        x: A non-negative tensor.

    Returns:
        tensor: A tensor normalized by it's sum.

    """
    y = EPSILON + x
    return x/numpy.sum(y)

def norm(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
    """
    Return the L2 norm of a tensor.

    Args:
        x: A tensor.
        axis (optional): the axis for taking the norm
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is none:
            float: The L2 norm of the tensor
               (i.e., the sqrt of the sum of the squared elements).
        else:
            tensor: The L2 norm along the specified axis.

    """
    if axis is None:
        return numpy.linalg.norm(x)
    else:
        if keepdims:
            return numpy.expand_dims(numpy.linalg.norm(x, axis=axis), axis)
        else:
            return numpy.linalg.norm(x, axis=axis)

def tmax(x: T.Tensor, axis: int=None, keepdims: bool=False)-> T.FloatingPoint:
    """
    Return the elementwise maximum of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the maximum.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            float: The overall maximum of the elements in the tensor
        else:
            tensor: The maximum of the tensor along the specified axis.

    """
    return numpy.max(x, axis=axis, keepdims=keepdims)

def tmin(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
    """
    Return the elementwise minimum of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the minimum.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            float: The overall minimum of the elements in the tensor
        else:
            tensor: The minimum of the tensor along the specified axis.

    """
    return numpy.min(x, axis=axis, keepdims=keepdims)

def mean(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
    """
    Return the mean of the elements of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the mean.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            float: The overall mean of the elements in the tensor
        else:
            tensor: The mean of the tensor along the specified axis.

    """
    return numpy.mean(x, axis=axis, keepdims=keepdims)

def var(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
    """
    Return the variance of the elements of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the variance.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            float: The overall variance of the elements in the tensor
        else:
            tensor: The variance of the tensor along the specified axis.

    """
    return numpy.var(x, axis=axis, keepdims=keepdims, ddof=1)

def std(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
    """
    Return the standard deviation of the elements of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the standard deviation.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            float: The overall standard deviation of the elements in the tensor
        else:
            tensor: The standard deviation of the tensor along the specified axis.

    """
    return numpy.std(x, axis=axis, keepdims=keepdims, ddof=1)

def tsum(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
    """
    Return the sum of the elements of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the sum.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            float: The overall sum of the elements in the tensor
        else:
            tensor: The sum of the tensor along the specified axis.

    """
    return numpy.sum(x, axis=axis, keepdims=keepdims)

def tprod(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
    """
    Return the product of the elements of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the product.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            float: The overall product of the elements in the tensor
        else:
            tensor: The product of the tensor along the specified axis.

    """
    return numpy.prod(x, axis=axis, keepdims=keepdims)

def tany(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.Boolean:
    """
    Return True if any elements of the input tensor are true along the
    specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis of interest.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            bool: 'any' applied to all elements in the tensor
        else:
            tensor (of bools): 'any' applied to the elements in the tensor
                                along axis

    """
    return numpy.any(x, axis=axis, keepdims=keepdims)

def tall(x: T.Tensor, axis: int=None, keepdims: bool=False) -> T.Boolean:
    """
    Return True if all elements of the input tensor are true along the
    specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis of interest.
        keepdims (optional): If this is set to true, the dimension of the tensor
                             is unchanged. Otherwise, the reduced axis is removed
                             and the dimension of the array is 1 less.

    Returns:
        if axis is None:
            bool: 'all' applied to all elements in the tensor
        else:
            tensor (of bools): 'all' applied to the elements in the tensor
                                along axis

    """
    return numpy.all(x, axis=axis, keepdims=keepdims)

def equal(x: T.Tensor, y: T.Tensor) -> T.Boolean:
    """
    Elementwise if two tensors are equal.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of equality between x and y.

    """
    return numpy.equal(x, y)

def allclose(x: T.Tensor, y: T.Tensor,
             rtol: float=1e-05, atol: float=1e-08) -> bool:
    """
    Test if all elements in the two tensors are approximately equal.

    absolute(x - y) <= (atol + rtol * absolute(y))

    Args:
        x: A tensor.
        y: A tensor.
        rtol (optional): Relative tolerance.
        atol (optional): Absolute tolerance.

    returns:
        bool: Check if all of the elements in the tensors are approximately equal.

    """
    return numpy.allclose(x, y, rtol=rtol, atol=atol)

def not_equal(x: T.Tensor, y: T.Tensor) -> T.Boolean:
    """
    Elementwise test if two tensors are not equal.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of non-equality between x and y.

    """
    return numpy.not_equal(x, y)

def greater(x: T.Tensor, y: T.Tensor) -> T.Boolean:
    """
    Elementwise test if x > y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x > y.

    """
    return numpy.greater(x, y)

def greater_equal(x: T.Tensor, y: T.Tensor) -> T.Boolean:
    """
    Elementwise test if x >= y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x >= y.

    """
    return numpy.greater_equal(x, y)

def lesser(x: T.Tensor, y: T.Tensor) -> T.Boolean:
    """
    Elementwise test if x < y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x < y.

    """
    return numpy.less(x, y)

def lesser_equal(x: T.Tensor, y: T.Tensor) -> T.Boolean:
    """
    Elementwise test if x <= y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x <= y.

    """
    return numpy.less_equal(x, y)

def maximum(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Elementwise maximum of two tensors.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor: Elementwise maximum of x and y.

    """
    return numpy.maximum(x, y)

def minimum(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Elementwise minimum of two tensors.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor: Elementwise minimum of x and y.

    """
    return numpy.minimum(x, y)

def argmax(x: T.Tensor, axis: int) -> T.Tensor:
    """
    Compute the indices of the maximal elements in x along the specified axis.

    Args:
        x: A tensor:
        axis: The axis of interest.

    Returns:
        tensor (of ints): Indices of the maximal elements in x along the
        specified axis.

    """
    return numpy.argmax(x, axis=axis)

def argmin(x: T.Tensor, axis: int) -> T.Tensor:
    """
    Compute the indices of the minimal elements in x along the specified axis.

    Args:
        x: A tensor:
        axis: The axis of interest.

    Returns:
        tensor (of ints): Indices of the minimum elements in x along the
        specified axis.

    """
    return numpy.argmin(x, axis=axis)

def dot(a: T.Tensor, b: T.Tensor) -> T.FloatingPoint:
    """
    Compute the matrix/dot product of tensors a and b.

    Vector-Vector:
        \sum_i a_i b_i

    Matrix-Vector:
        \sum_j a_ij b_j

    Matrix-Matrix:
        \sum_j a_ij b_jk

    Args:
        a: A tensor.
        b: A tensor:

    Returns:
        if a and b are 1-dimensions:
            float: the dot product of vectors a and b
        else:
            tensor: the matrix product of tensors a and b

    """
    return numpy.dot(a, b)

def outer(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Compute the outer product of vectors x and y.

    mat_{ij} = x_i * y_j

    Args:
        x: A vector (i.e., a 1D tensor).
        y: A vector (i.e., a 1D tensor).

    Returns:
        tensor: Outer product of vectors x and y.

    """
    return numpy.outer(x,y)


class BroadcastError(ValueError):
    """
    BroadcastError exception:

    Args: None

    """

    pass

def broadcast(vec: T.Tensor, matrix: T.Tensor) -> T.Tensor:
    """
    Broadcasts vec into the shape of matrix following numpy rules:

    vec ~ (N, 1) broadcasts to matrix ~ (N, M)
    vec ~ (1, N) and (N,) broadcast to matrix ~ (M, N)

    Args:
        vec: A vector (either flat, row, or column).
        matrix: A matrix (i.e., a 2D tensor).

    Returns:
        tensor: A tensor of the same size as matrix containing the elements
                of the vector.

    Raises:
        BroadcastError

    """
    try:
        return numpy.broadcast_to(vec, shape(matrix))
    except ValueError:
        raise BroadcastError('cannot broadcast vector of dimension {} \
onto matrix of dimension {}'.format(shape(vec), shape(matrix)))

def add(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """
    Add tensor a to tensor b using broadcasting.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        tensor: a + b

    """
    if shape(a) == shape(b):
        # no broadcasting necessary
        return a + b
    else:
        return broadcast(a, b) + b

def subtract(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """
    Subtract tensor a from tensor b using broadcasting.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        tensor: b - a

    """
    if shape(a) == shape(b):
        # no broadcasting necessary
        return b - a
    else:
        return b - broadcast(a, b)

def multiply(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """
    Multiply tensor b with tensor a using broadcasting.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        tensor: a * b

    """
    if shape(a) == shape(b):
        # no broadcasting necessary
        return a * b
    else:
        return broadcast(a, b) * b

def divide(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """
    Divide tensor b by tensor a using broadcasting.

    Args:
        a: A tensor (non-zero)
        b: A tensor

    Returns:
        tensor: b / a

    """
    if shape(a) == shape(b):
        # no broadcasting necessary
        return b / a
    else:
        return b / broadcast(a, b)

def affine(a: T.Tensor, b: T.Tensor, W: T.Tensor) -> T.Tensor:
    """
    Evaluate the affine transformation a + W b.

    a ~ vector, b ~ vector, W ~ matrix:
    a_i + \sum_j W_ij b_j

    a ~ matrix, b ~ matrix, W ~ matrix:
    a_ij + \sum_k W_ik b_kj

    Args:
        a: A tensor (1 or 2 dimensional).
        b: A tensor (1 or 2 dimensional).
        W: A tensor (2 dimensional).

    Returns:
        tensor: Affine transformation a + W b.

    """
    return a + numpy.dot(W,b)

def quadratic(a: T.Tensor, b: T.Tensor, W: T.Tensor) -> T.Tensor:
    """
    Evaluate the quadratic form a W b.

    a ~ vector, b ~ vector, W ~ matrix:
    \sum_ij a_i W_ij b_j

    a ~ matrix, b ~ matrix, W ~ matrix:
    \sum_kl a_ik W_kl b_lj

    Args:
        a: A tensor:
        b: A tensor:
        W: A tensor:

    Returns:
        tensor: Quadratic function a W b.

    """
    return numpy.dot(a, numpy.dot(W, b))

def inv(mat: T.Tensor) -> T.Tensor:
    """
    Compute matrix inverse.

    Args:
        mat: A square matrix.

    Returns:
        tensor: The matrix inverse.

    """
    return numpy.linalg.inv(mat)

def batch_dot(vis: T.Tensor, W: T.Tensor, hid: T.Tensor, axis: int=1) -> T.Tensor:
    """
    Let v by a L x N matrix where each row v_i is a visible vector.
    Let h be a L x M matrix where each row h_i is a hidden vector.
    And, let W be a N x M matrix of weights.
    Then, batch_dot(v,W,h) = \sum_i v_i^T W h_i

    The actual computation is performed with a vectorized expression.

    Args:
        vis: A tensor.
        W: A tensor.
        hid: A tensor.
        axis (optional): Axis of interest

    Returns:
        tensor: A vector.

    """
    return (numpy.dot(vis, W) * hid).sum(axis).astype(numpy.float32)

def batch_outer(vis: T.Tensor, hid: T.Tensor) -> T.Tensor:
    """
    Let v by a L x N matrix where each row v_i is a visible vector.
    Let h be a L x M matrix where each row h_i is a hidden vector.
    Then, batch_outer(v, h) = \sum_i v_i h_i^T
    Returns an N x M matrix.

    The actual computation is performed with a vectorized expression.

    Args:
        vis: A tensor.
        hid: A tensor:

    Returns:
        tensor: A matrix.

    """
    return numpy.dot(vis.T, hid)

def repeat(tensor: T.Tensor, n: int) -> T.Tensor:
    """
    Repeat tensor n times along the first axis.

    Args:
        tensor: A vector (i.e., 1D tensor).
        n: The number of repeats.

    Returns:
        tensor: A vector created from many repeats of the input tensor.

    """
    # current implementation only works for vectors
    assert ndim(tensor) == 1
    return numpy.repeat(tensor, n, axis=0)

def stack(tensors: T.Iterable[T.Tensor], axis: int) -> T.Tensor:
    """
    Stack tensors along the specified axis.

    Args:
        tensors: A list of tensors.
        axis: The axis the tensors will be stacked along.

    Returns:
        tensor: Stacked tensors from the input list.

    """
    return numpy.stack(tensors, axis=axis)

def hstack(tensors: T.Iterable[T.Tensor]) -> T.Tensor:
    """
    Concatenate tensors along the first axis.

    Args:
        tensors: A list of tensors.

    Returns:
        tensor: Tensors stacked along axis=1.

    """
    return numpy.hstack(tensors)

def vstack(tensors:  T.Iterable[T.Tensor]) -> T.Tensor:
    """
    Concatenate tensors along the zeroth axis.

    Args:
        tensors: A list of tensors.

    Returns:
        tensor: Tensors stacked along axis=0.

    """
    return numpy.vstack(tensors)

def trange(start: int, end: int, step: int=1) -> T.Tensor:
    """
    Generate a tensor like a python range.

    Args:
        start: The start of the range.
        end: The end of the range.
        step: The step of the range.

    Returns:
        tensor: A vector ranging from start to end in increments
                of step. Cast to float rather than int.

    """
    return numpy.arange(start, end, step, dtype=numpy.float32)

@jit('float32(float32[:],float32[:])',nopython=True)
def squared_euclidean_distance(a, b):
    """
    Compute the squared euclidean distance between two vectors.

    Args:
        a: A vector (i.e., 1D tensor).
        b: A vector (i.e., 1D tensor).

    Returns:
        float: Squared euclidean distance between a and b.

    """
    result = numpy.float32(0.0)
    for i in range(len(a)):
        result += (a[i] - b[i])**2
    return result

@jit('float32(float32[:],float32[:])',nopython=True)
def euclidean_distance(a, b):
    """
    Compute the euclidean distance between two vectors.

    Args:
        a: A vector (i.e., 1D tensor).
        b: A vector (i.e., 1D tensor).

    Returns:
        float: Euclidean distance between a and b.

    """
    return math.sqrt(squared_euclidean_distance(a, b))

@jit('float32[:,:](float32[:,:], int16, boolean)',nopython=True)
def resample(x, n, replace=True):
    """
    Resample a tensor along the zeroth axis.

    Args:
        x: A tensor.
        n: Number of samples to take.
        replace (optional): Sample with replacement if True.

    Returns:
        tensor: Sample of size n from x along axis=0.

    """
    index = numpy.random.choice(numpy.arange(len(x)),size=n,replace=replace)
    return x[index]

@jit('float32(float32[:,:],float32[:,:], int16)',nopython=True)
def fast_energy_distance(minibatch, samples, downsample=100):
    """
    Compute an approximate energy distance between two tensors.

    Args:
        minibatch: A 2D tensor.
        samples: A 2D tensor.

    Returns:
        float: Approximate energy distance.

    Szekely, G.J. (2002)
    E-statistics: The Energy of Statistical Samples.
    Technical Report BGSU No 02-16.

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
