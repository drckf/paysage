import numpy
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
    return T.Float(scalar)

def cast_float(tensor: T.Tensor) -> T.Tensor:
    """
    Cast tensor to a float tensor.

    Notes:
        If tensor is already float, no copy is made.

    Args:
        tensor: A tensor.

    Returns:
        tensor: copy of tensor converted to floating point.

    """
    return tensor.astype(T.Float, copy=False)

def float_tensor(tensor: T.FloatConstructable) -> T.Tensor:
    """
    Construct a float tensor.
    This will always copy the data in tensor.

    Args:
        tensor: A float tensor or list of floats.

    Returns:
        tensor: Constructed float tensor.

    """
    return numpy.array(tensor, dtype=T.Float)

def cast_long(tensor: T.Tensor) -> T.Tensor:
    """
    Cast tensor to an long int tensor.
    Notes:
        If tensor is already long int, no copy is made.

    Args:
        tensor: A tensor.

    Returns:
        tensor: Copy of tensor converted to long int.

    """
    return tensor.astype(T.Long, copy=False)

def long_tensor(tensor: T.LongConstructable) -> T.Tensor:
    """
    Construct a long int tensor.
    This will always copy the data in tensor.

    Args:
        tensor: A long tensor or list of longs.

    Returns:
        tensor: Tensor converted to long int.

    """
    return numpy.array(tensor, dtype=T.Long)

def to_numpy_array(tensor: T.Tensor) -> T.Tensor:
    """
    Return tensor as a numpy array.

    Args:
        tensor: A tensor.

    Returns:
        tensor: Tensor converted to a numpy array.

    """
    return tensor

def from_numpy_array(tensor: T.Tensor) -> T.NumpyTensor:
    """
    Construct a tensor from a numpy array. A noop.

    Args:
        tensor: A numpy ndarray

    Returns:
        tensor: Tensor converted from ndarray.

    """
    return tensor

def copy_tensor(tensor: T.Tensor) -> T.Tensor:
    """
    Copy a tensor.

    Args:
        tensor

    Returns:
        copy of tensor

    """
    return tensor.astype(tensor.dtype)

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

def zeros(shape: T.Tuple[int], dtype: T.Dtype = T.Float) -> T.Tensor:
    """
    Return a tensor of a specified shape filled with zeros.

    Args:
        shape: The shape of the desired tensor.

    Returns:
        tensor: A tensor of zeros with the desired shape.

    """
    return numpy.zeros(shape, dtype=dtype)

def zeros_like(tensor: T.Tensor) -> T.Tensor:
    """
    Return a tensor of zeros with the same shape as the input tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor of zeros with the same shape.

    """
    return zeros(shape(tensor))

def ones(shape: T.Tuple[int], dtype: T.Dtype = T.Float) -> T.Tensor:
    """
    Return a tensor of a specified shape filled with ones.

    Args:
        shape: The shape of the desired tensor.

    Returns:
        tensor: A tensor of ones with the desired shape.

    """
    return numpy.ones(shape, dtype=dtype)

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
    return numpy.identity(n, dtype=T.Float)

def fill_diagonal_(mat: T.Tensor, val: T.Scalar) -> T.Tensor:
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

def scatter_(mat: T.Tensor, inds: T.Tensor, val: T.Scalar) -> T.Tensor:
    """
    Assign a value a specific points in a matrix.
    Iterates along the rows of mat,
    successively assigning val to column indices given by inds.

    Note:
        Modifies mat in place.

    Args:
        mat: A tensor.
        inds: The indices
        val: The value to insert
    """
    mat[range(len(mat)), inds] = val

def index_select(mat: T.Tensor, index: T.Tensor, dim: int = 0) -> T.Tensor:
    """
    Select the specified indices of a tensor along dimension dim.
    For example, dim = 1 is equivalent to mat[:, index] in numpy.

    Args:
        mat (tensor (num_samples, num_units))
        index (tensor; 1 -dimensional)
        dim (int)

    Returns:
        if dim == 0:
            mat[index, :]
        if dim == 1:
            mat[:, index]

    """
    if dim == 0:
        return mat[index, :]
    elif dim == 1:
        return mat[:, index]

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

def tclip(tensor: T.Tensor, a_min: T.Tensor=None,
         a_max: T.Tensor=None) -> T.Tensor:
    """
    Return a tensor with its values clipped element-wise between a_min and a_max tensors.
    The implementation is identical to clip.

    Args:
        tensor: A tensor.
        a_min (optional tensor): The desired lower bound on the elements of the tensor.
        a_max (optional tensor): The desired upper bound on the elements of the tensor.

    Returns:
        tensor: A new tensor with its values clipped between a_min and a_max.

    """
    return tensor.clip(a_min, a_max)

def clip_(tensor: T.Tensor, a_min: T.Scalar=None,
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

def tclip_(tensor: T.Tensor, a_min: T.Tensor=None,
                 a_max: T.Tensor=None) -> None:
    """
    Clip the values of a tensor elementwise between a_min and a_max tensors.
    The implementation is identical to clip_

    Note:
        Modifies tensor in place.

    Args:
        tensor: A tensor.
        a_min (optional tensor): The desired lower bound on the elements of the tensor.
        a_max (optional tessor): The desired upper bound on the elements of the tensor.

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

def tfloor(tensor: T.Tensor) -> T.Tensor:
    """
    Return a tensor with floored elements.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor rounded down to the next integer (still floating point).

    """
    return numpy.floor(tensor)

def tceil(tensor: T.Tensor) -> T.Tensor:
    """
    Return a tensor with ceilinged elements.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor rounded up to the next integer (still floating point).

    """
    return numpy.ceil(tensor)

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

def mix(w: T.Tensor, x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Compute a weighted average of two matrices (x and y) and return the result.
    Multilinear interpolation.

    Note:
        Modifies x in place.

    Args:
        w: The mixing coefficient tensor between 0 and 1 .
        x: A tensor.
        y: A tensor:

    Returns:
        tensor = w * x + (1-w) * y

    """
    return ne.evaluate('w*x + (1-w)*y')

def mix_(w: T.Tensor, x: T.Tensor, y: T.Tensor) -> None:
    """
    Compute a weighted average of two matrices (x and y) and store the results in x.
    Useful for keeping track of running averages during training.

    x <- w * x + (1-w) * y

    Note:
        Modifies x in place.

    Args:
        w: The mixing coefficient tensor between 0 and 1 .
        x: A tensor.
        y: A tensor:

    Returns:
        None

    """
    ne.evaluate('w*x + (1-w)*y', out=x)

def square_mix_(w: T.Tensor, x: T.Tensor, y: T.Tensor) -> None:
    """
    Compute a weighted average of two matrices (x and y^2) and store the results in x.
    Useful for keeping track of running averages of squared matrices during training.

    x <- w x + (1-w) * y**2

    Note:
        Modifies x in place.

    Args:
        w: The mixing coefficient tensor between 0 and 1.
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
    z = T.EPSILON + y
    return ne.evaluate('x/sqrt(z)')

def normalize(x: T.Tensor) -> T.Tensor:
    """
    Divide x by it's sum.

    Args:
        x: A non-negative tensor.

    Returns:
        tensor: A tensor normalized by it's sum.

    """
    y = T.EPSILON + x
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
    return numpy.linalg.norm(x, axis=axis, keepdims=keepdims)

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

def mean(x: T.Tensor, axis: int = None, keepdims: bool = False) -> T.FloatingPoint:
    """
    Return the mean of the elements of a tensor along the specified axis.

    Args:
        x: A float or tensor of rank=2.
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

def center(x: T.Tensor, axis: int=0) -> T.Tensor:
    """
    Remove the mean along axis.

    Args:
        tensor (num_samples, num_units): the array to center
        axis (int; optional): the axis to center along

    Returns:
        tensor (num_samples, num_units)

    """
    return subtract(mean(x, axis=axis, keepdims=True), x)

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

def cov(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Compute the cross covariance between tensors x and y.

    Args:
        x (tensor (num_samples, num_units_x))
        y (tensor (num_samples, num_units_y))

    Returns:
        tensor (num_units_x, num_units_y)

    """
    num_samples = len(x) - 1
    return batch_outer(center(x), center(y)) / num_samples

def corr(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Compute the cross correlation between tensors x and y.

    Args:
        x (tensor (num_samples, num_units_x))
        y (tensor (num_samples, num_units_y))

    Returns:
        tensor (num_units_x, num_units_y)

    """
    covariance = cov(x, y)
    std_x = std(x, axis=0) + T.EPSILON
    std_y = std(y, axis=0) + T.EPSILON
    return divide(outer(std_x, std_y), covariance)

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

def sort(x: T.Tensor, axis: int = None) -> T.Tensor:
    """
    Sort a tensor along the specied axis.

    Args:
        x: A tensor:
        axis: The axis of interest.

    Returns:
        tensor (of floats): sorted tensor

    """
    return numpy.sort(x, axis=axis)

def argsort(x: T.Tensor, axis: int = None) -> T.Tensor:
    """
    Get the indices of a sorted tensor.
    If axis=None this flattens x

    Args:
        x: A tensor:
        axis: The axis of interest.

    Returns:
        tensor (of ints): indices of sorted tensor

    """
    return numpy.argsort(x, axis=axis)

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
    r = numpy.argmax(x, axis=axis)
    if numpy.isscalar(r):
        return cast_long(numpy.array([r]))
    return r

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
    r = numpy.argmin(x, axis=axis)
    if numpy.isscalar(r):
        return cast_long(numpy.array([r]))
    return r

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
    return a + b

def add_(a: T.Tensor, b: T.Tensor) -> None:
    """
    Add tensor a to tensor b using broadcasting.

    Notes:
        Modifies b in place.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        None

    """
    ne.evaluate("a + b", out=b)

def subtract(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """
    Subtract tensor a from tensor b using broadcasting.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        tensor: b - a

    """
    return b - a

def subtract_(a: T.Tensor, b: T.Tensor) -> None:
    """
    Subtract tensor a from tensor b using broadcasting.

    Notes:
        Modifies b in place.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        None

    """
    ne.evaluate("b - a", out=b)

def multiply(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """
    Multiply tensor b with tensor a using broadcasting.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        tensor: a * b

    """
    return a * b

def multiply_(a: T.Tensor, b: T.Tensor) -> None:
    """
    Multiply tensor b with tensor a using broadcasting.

    Notes:
        Modifies b in place.

    Args:
        a: A tensor
        b: A tensor

    Returns:
        None

    """
    ne.evaluate("a * b", out=b)

def divide(a: T.Tensor, b: T.Tensor) -> T.Tensor:
    """
    Divide tensor b by tensor a using broadcasting.

    Args:
        a: A tensor (non-zero)
        b: A tensor

    Returns:
        tensor: b / a

    """
    return b / a

def divide_(a: T.Tensor, b: T.Tensor) -> None:
    """
    Divide tensor b by tensor a using broadcasting.

    Notes:
        Modifies b in place.

    Args:
        a: A tensor (non-zero)
        b: A tensor

    Returns:
        tensor: b / a

    """
    ne.evaluate("b / a", out=b)

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
    if len(mat) > 1:
        return numpy.linalg.inv(mat)
    return numpy.reciprocal(mat)

def pinv(mat: T.Tensor) -> T.Tensor:
    """
    Compute matrix pseudoinverse.

    Args:
        mat: A square matrix.

    Returns:
        tensor: The matrix pseudoinverse.

    """
    return numpy.linalg.pinv(mat)

def qr(mat: T.Tensor) -> T.Tuple[T.Tensor]:
    """
    Compute the QR decomposition of a matrix.
    The QR decomposition factorizes a matrix A into a product
    A = QR of an orthonormal matrix Q and an upper triangular matrix R.
    Provides an orthonormalization of the columns of the matrix.

    Args:
        mat: A matrix.

    Returns:
        (Q, R): Tuple of tensors.

    """
    return numpy.linalg.qr(mat)

def svd(mat: T.Tensor) -> T.Tuple[T.Tensor]:
    """
    Compute the Singular Value decomposition of a matrix
    A = U S V^T

    Args:
        mat: A matrix.

    Returns:
        (U, S, V): Tuple of tensors.

    """
    u, s, v = numpy.linalg.svd(mat, full_matrices=False)
    return u, s, v.T

def matrix_sqrt(mat: T.Tensor) -> T.Tensor:
    """
    Compute the matrix square root using an SVD

    Args:
        mat: A square matrix.

    Returns:
        logdet: The logarithm of the matrix determinant.

    """
    u, s, v = svd(mat)
    return numpy.dot(u * numpy.sqrt(s), v.T)

def logdet(mat: T.Tensor) -> float:
    """
    Compute the logarithm of the determinant of a square matrix.

    Args:
        mat: A square matrix.

    Returns:
        logdet: The logarithm of the matrix determinant.

    """
    return numpy.linalg.slogdet(mat)[1]

def batch_dot(a: T.Tensor, b: T.Tensor, axis: int=1) -> T.Tensor:
    """
    Compute the dot product of vectors batch-wise.
    Let a be an L x N matrix where each row a_i is a vector.
    Let b be an L x N matrix where each row b_i is a vector.
    Then batch_dot(a, b) = \sum_j a_ij * b_ij
    One can choose the axis to sum along with the axis argument.

    Args:
        a: A tensor.
        b: A tensor.
        axis (int): The axis to dot along.

    Returns:
        tensor: A tensor.

    """
    return (a * b).sum(axis=axis)

def batch_quadratic(vis: T.Tensor, W: T.Tensor, hid: T.Tensor, axis: int=1) -> T.Tensor:
    """
    Let v by an L x N matrix where each row v_i is a visible vector.
    Let h be an L x M matrix where each row h_i is a hidden vector.
    And, let W be a N x M matrix of weights.
    Then, batch_quadratic(v,W,h) = \sum_i v_i^T W h_i

    The actual computation is performed with a vectorized expression.

    Args:
        vis: A tensor.
        W: A tensor.
        hid: A tensor.
        axis (optional): Axis of interest

    Returns:
        tensor: A vector.

    """
    return (numpy.dot(vis, W) * hid).sum(axis).astype(T.Float)

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

def trange(start: int, end: int, step: int=1, dtype: T.Dtype = None) -> T.Tensor:
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
    return numpy.arange(start, end, step, dtype=dtype)

def cumsum(x: T.Tensor, axis: int = 0) -> T.Tensor:
    """
    Return the cumulative sum of elements of a tensor along the specified axis.

    Args:
        x: A float or tensor.
        axis (optional): The axis for taking the sum.

    Returns:
        tensor: the cumulative sum of elements of the tensor along the specified axis.

    """
    return x.cumsum(axis=axis)

def is_tensor(x: T.FloatingPoint) -> bool:
    """
    Test if x is a tensor.

    Args:
        x (float or tensor)

    Returns:
        bool

    """
    try:
        shape(x)
        return True
    except Exception:
        return False

def logical_not(x: T.Tensor) -> T.Tensor:
    """
    Invert a logical array (True -> False, False -> True).

    Args:
        x (tensor)

    Returns:
        tensor

    """
    return numpy.logical_not(x)

def logical_and(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Compute the elementwise logical and on two tensors

    Args:
        x (tensor)
        y (tensor)

    Returns:
        tensor

    """
    return numpy.logical_and(x, y)

def logical_or(x: T.Tensor, y: T.Tensor) -> T.Tensor:
    """
    Compute the elementwise logical or on two tensors

    Args:
        x (tensor)
        y (tensor)

    Returns:
        tensor

    """
    return numpy.logical_or(x, y)
