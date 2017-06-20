import numpy, torch
from . import typedef as T

def float_scalar(scalar: T.Scalar) -> float:
    """
    Cast scalar to a float.

    Args:
        scalar: A scalar quantity:

    Returns:
        float: Scalar converted to floating point.

    """
    return float(scalar)

EPSILON = float_scalar(numpy.finfo(numpy.float32).eps)

def float_tensor(tensor: T.Tensor) -> T.FloatTensor:
    """
    Cast tensor to a float tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: Tensor converted to floating point.

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

    Args:
        tensor: A tensor.

    Returns:
        tensor: Tensor converted to a numpy array.

    """
    try:
        return tensor.numpy()
    except Exception:
        return numpy.array(tensor)

def shape(tensor: T.TorchTensor) -> T.Tuple[int]:
    """
    Return a tuple with the shape of the tensor.

    Args:
        tensor: A tensor:

    Returns:
        tuple: A tuple of integers describing the shape of the tensor.

    """
    return tuple(tensor.size())

def ndim(tensor: T.TorchTensor) -> int:
    """
    Return the number of dimensions of a tensor.

    Args:
        tensor: A tensor:

    Returns:
        int: The number of dimensions of the tensor.

    """
    return tensor.ndimension()

def num_elements(tensor: T.TorchTensor) -> int:
    """
    Return the number of elements in a tensor.

    Args:
        tensor: A tensor:

    Returns:
        int: The number of elements in the tensor.

    """
    return numpy.prod(shape(tensor))

def transpose(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return the transpose of a tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: The transpose (exchange of rows and columns) of the tensor.

    """
    return tensor.t()

def zeros(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Return a tensor of a specified shape filled with zeros.

    Args:
        shape: The shape of the desired tensor.

    Returns:
        tensor: A tensor of zeros with the desired shape.

    """
    return torch.zeros(shape)

def zeros_like(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return a tensor of zeros with the same shape as the input tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor of zeros with the same shape.

    """
    return zeros(shape(tensor))

def ones(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Return a tensor of a specified shape filled with ones.

    Args:
        shape: The shape of the desired tensor.

    Returns:
        tensor: A tensor of ones with the desired shape.

    """
    return torch.ones(shape)

def ones_like(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return a tensor of ones with the same shape as the input tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor with the same shape.

    """
    return ones(shape(tensor))

def diagonal_matrix(vec: T.TorchTensor) -> T.TorchTensor:
    """
    Return a matrix with vec along the diagonal.

    Args:
        vec: A vector (i.e., 1D tensor).

    Returns:
        tensor: A matrix with the elements of vec along the diagonal,
                and zeros elsewhere.

    """
    return torch.diag(vec)

def diag(mat: T.TorchTensor) -> T.TorchTensor:
    """
    Return the diagonal elements of a matrix.

    Args:
        mat: A tensor.

    Returns:
        tensor: A vector (i.e., 1D tensor) containing the diagonal
                elements of mat.

    """
    return mat.diag()

def identity(n: int) -> T.FloatTensor:
    """
    Return the n-dimensional identity matrix.

    Args:
        n: The desired size of the tensor.

    Returns:
        tensor: The n x n identity matrix with ones along the diagonal
                and zeros elsewhere.

    """
    return torch.eye(n)

def fill_diagonal(mat: T.FloatTensor, val: T.Scalar) -> None:
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
    for i in range(len(mat)):
        mat[i,i] = val

def sign(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Return the elementwise sign of a tensor.

    Args:
        tensor: A tensor.

    Returns:
        tensor: The sign of the elements in the tensor.

    """
    return tensor.sign()

def clip(tensor: T.FloatTensor,
         a_min: T.Scalar = -numpy.inf,
         a_max: T.Scalar = numpy.inf) -> T.FloatTensor:
    """
    Return a tensor with its values clipped between a_min and a_max.

    Args:
        tensor: A tensor.
        a_min (optional): The desired lower bound on the elements of the tensor.
        a_max (optional): The desired upper bound on the elements of the tensor.

    Returns:
        tensor: A new tensor with its values clipped between a_min and a_max.

    """
    return tensor.clamp(a_min, a_max)

def clip_inplace(tensor: T.FloatTensor,
                 a_min: T.Scalar = -numpy.inf,
                 a_max: T.Scalar = numpy.inf) -> None:
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
    return torch.clamp(tensor, a_min, a_max, out=tensor)

def tround(tensor: T.FloatTensor) -> T.FloatTensor:
    """
    Return a tensor with rounded elements.

    Args:
        tensor: A tensor.

    Returns:
        tensor: A tensor rounded to the nearest integer (still floating point).

    """
    return tensor.round()

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
        return tensor.view(int(numpy.prod(shape(tensor))))
    except AttributeError:
        return tensor

def reshape(tensor: T.FloatTensor,
            newshape: T.Tuple[int]) -> T.FloatTensor:
    """
    Return tensor with a new shape.

    Args:
        tensor: A tensor.
        newshape: The desired shape.

    Returns:
        tensor: A tensor with the desired shape.

    """
    return tensor.view(*newshape)

def unsqueeze(tensor: T.Tensor, axis: int) -> T.Tensor:
    """
    Return tensor with a new axis inserted.

    Args:
        tensor: A tensor.
        axis: The desired axis.

    Returns:
        tensor: A tensor with the new axis inserted.

    """
    return torch.unsqueeze(tensor, axis)

def dtype(tensor: T.FloatTensor) -> type:
    """
    Return the type of the tensor.

    Args:
        tensor: A tensor.

    Returns:
        type: The type of the elements in the tensor.

    """
    return tensor.type()

def mix_inplace(w: T.Scalar,
                x: T.FloatTensor,
                y: T.FloatTensor) -> None:
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
    x.mul_(w)
    x.add_(y.mul(1-w))

def square_mix_inplace(w: T.Scalar,
                       x: T.FloatTensor,
                       y: T.FloatTensor) -> None:
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
    x.mul_(w)
    x.add_(y.mul(y).mul(1-w))

def sqrt_div(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise division of x by sqrt(y).

    Args:
        x: A tensor:
        y: A non-negative tensor.

    Returns:
        tensor: Elementwise division of x by sqrt(y).

    """
    return x.div(torch.sqrt(EPSILON + y))

def normalize(x: T.FloatTensor) -> T.FloatTensor:
    """
    Divide x by it's sum.

    Args:
        x: A non-negative tensor.

    Returns:
        tensor: A tensor normalized by it's sum.

    """
    return x.div(torch.sum(EPSILON + x))

def norm(x: T.FloatTensor,  axis: int=None, keepdims: bool=False) -> T.FloatingPoint:
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
        return x.norm()
    else:
        if keepdims:
            return x.norm(2, axis)
        else:
            return flatten(x.norm(2, axis))

def tmax(x: T.FloatTensor,
         axis: int = None,
         keepdims: bool = False) -> T.FloatingPoint:
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
    if axis is not None:
        tmp = x.prod(dim=axis)
        if keepdims:
            return tmp
        else:
            return flatten(tmp)
    else:
        return x.prod()

def tany(x: T.Tensor,
         axis: int = None,
         keepdims: bool = False) -> T.Boolean:
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
            tensor (of bytes): 'any' applied to the elements in the tensor
                                along axis

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
            tensor (of bytes): 'all' applied to the elements in the tensor
                                along axis

    """
    tmp = tmin(x.ne(0), axis=axis)
    if keepdims:
        return tmp
    else:
        return flatten(tmp)

def equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test for if two tensors are equal.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of equality between x and y.

    """
    return torch.eq(x, y)

def allclose(x: T.FloatTensor,
             y: T.FloatTensor,
             rtol: float = 1e-05,
             atol: float = 1e-08) -> bool:
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
    return tall(torch.abs(x - y).le((atol + rtol * torch.abs(y))))

def not_equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if two tensors are not equal.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bytes): Elementwise test of non-equality between x and y.

    """
    return torch.ne(x, y)

def greater(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x > y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x > y.

    """
    return torch.gt(x, y)

def greater_equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x >= y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x >= y.

    """
    return torch.ge(x, y)

def lesser(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x < y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x < y.

    """
    return torch.lt(x, y)

def lesser_equal(x: T.FloatTensor, y: T.FloatTensor) -> T.ByteTensor:
    """
    Elementwise test if x <= y.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor (of bools): Elementwise test of x <= y.

    """
    return torch.le(x, y)

def maximum(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise maximum of two tensors.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor: Elementwise maximum of x and y.

    """
    return torch.max(x, y)

def minimum(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise minimum of two tensors.

    Args:
        x: A tensor.
        y: A tensor.

    Returns:
        tensor: Elementwise minimum of x and y.

    """
    return torch.min(x, y)

def argmax(x: T.FloatTensor, axis: int) -> T.LongTensor:
    """
    Compute the indices of the maximal elements in x along the specified axis.

    Args:
        x: A tensor:
        axis: The axis of interest.

    Returns:
        tensor (of ints): Indices of the maximal elements in x along the
                          specified axis.

    """
    # needs flatten because numpy argmax always returns a 1-D array
    return flatten(x.max(dim=axis)[1])

def argmin(x: T.FloatTensor, axis: int = None) -> T.LongTensor:
    """
    Compute the indices of the minimal elements in x along the specified axis.

    Args:
        x: A tensor:
        axis: The axis of interest.

    Returns:
        tensor (of ints): Indices of the minimum elements in x along the
                          specified axis.

    """
    # needs flatten because numpy argmin always returns a 1-D array
    return flatten(x.min(dim=axis)[1])

def dot(a: T.FloatTensor, b: T.FloatTensor) -> T.FloatingPoint:
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
    return a @ b

def outer(x: T.FloatTensor, y: T.FloatTensor) -> T.FloatTensor:
    """
    Compute the outer product of vectors x and y.

    mat_{ij} = x_i * y_j

    Args:
        x: A vector (i.e., a 1D tensor).
        y: A vector (i.e., a 1D tensor).

    Returns:
        tensor: Outer product of vectors x and y.

    """
    return torch.ger(x, y)

class BroadcastError(ValueError):
    """
    BroadcastError exception:

    Args: None

    """
    pass

def broadcast(vec: T.FloatTensor, matrix: T.FloatTensor) -> T.FloatTensor:
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
        if ndim(vec) == 1:
            if ndim(matrix) == 1:
                return vec
            return vec.unsqueeze(0).expand(matrix.size(0), matrix.size(1))
        else:
            return vec.expand(matrix.size(0), matrix.size(1))
    except ValueError:
        raise BroadcastError('cannot broadcast vector of dimension {} \
              onto matrix of dimension {}'.format(shape(vec), shape(matrix)))

def add(a: T.FloatTensor, b: T.FloatTensor) -> T.FloatTensor:
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

def add_(a: T.FloatTensor, b: T.FloatTensor) -> None:
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
    if shape(a) == shape(b):
        # no broadcasting necessary
        b.add_(a)
    else:
        b.add_(broadcast(a, b))

def subtract(a: T.FloatTensor, b: T.FloatTensor) -> T.FloatTensor:
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

def subtract_(a: T.FloatTensor, b: T.FloatTensor) -> None:
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
    if shape(a) == shape(b):
        # no broadcasting necessary
         b.sub_(a)
    else:
         b.sub_(broadcast(a, b))

def multiply(a: T.FloatTensor, b: T.FloatTensor) -> T.FloatTensor:
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

def multiply_(a: T.FloatTensor, b: T.FloatTensor) -> None:
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
    if shape(a) == shape(b):
        # no broadcasting necessary
        b.mul_(a)
    else:
        b.mul_(broadcast(a, b))

def divide(a: T.FloatTensor, b: T.FloatTensor) -> T.FloatTensor:
    """
    Divide tensor b by tensor a using broadcasting.

    Args:
        a: A tensor (non-zero)
        b: A tensor

    Returns:
        tensor: a * b

    """
    if shape(a) == shape(b):
        # no broadcasting necessary
        return b / a
    else:
        return b / broadcast(a, b)

def divide_(a: T.FloatTensor, b: T.FloatTensor) -> None:
    """
    Divide tensor b by tensor a using broadcasting.

    Notes:
        Modifies b in place.

    Args:
        a: A tensor (non-zero)
        b: A tensor

    Returns:
        None

    """
    if shape(a) == shape(b):
        # no broadcasting necessary
        b.div_(a)
    else:
        b.div_(broadcast(a, b))

def affine(a: T.FloatTensor,
           b: T.FloatTensor,
           W: T.FloatTensor) -> T.FloatTensor:
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
    return a @ W @ b

def inv(mat: T.FloatTensor) -> T.FloatTensor:
    """
    Compute matrix inverse.

    Args:
        mat: A square matrix.

    Returns:
        tensor: The matrix inverse.

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

    The actual computation is performed with a vectorized expression.

    Args:
        vis: A tensor.
        W: A tensor.
        hid: A tensor.
        axis (optional): Axis of interest

    Returns:
        tensor: A vector.

    """
    return tsum(dot(vis, W) * hid, axis)

def batch_outer(vis: T.FloatTensor, hid: T.FloatTensor) -> T.FloatTensor:
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
    return dot(transpose(vis), hid)

def repeat(tensor: T.FloatTensor, n: int) -> T.FloatTensor:
    """
    Repeat tensor n times along specified axis.

    Args:
        tensor: A vector (i.e., 1D tensor).
        n: The number of repeats.

    Returns:
        tensor: A vector created from many repeats of the input tensor.

    """
    # current implementation only works for vectors
    assert ndim(tensor) == 1
    return flatten(tensor.unsqueeze(1).repeat(1, n))

def stack(tensors: T.Iterable[T.FloatTensor], axis: int) -> T.FloatTensor:
    """
    Stack tensors along the specified axis.

    Args:
        tensors: A list of tensors.
        axis: The axis the tensors will be stacked along.

    Returns:
        tensor: Stacked tensors from the input list.

    """
    return torch.stack(tensors, dim=axis)

def hstack(tensors: T.Iterable[T.FloatTensor]) -> T.FloatTensor:
    """
    Concatenate tensors along the first axis.

    Args:
        tensors: A list of tensors.

    Returns:
        tensor: Tensors stacked along axis=1.

    """
    if ndim(tensors[0]) == 1:
        return torch.cat(tensors, 0)
    else:
        return torch.cat(tensors, 1)

def vstack(tensors: T.Iterable[T.FloatTensor]) -> T.FloatTensor:
    """
    Concatenate tensors along the zeroth axis.

    Args:
        tensors: A list of tensors.

    Returns:
        tensor: Tensors stacked along axis=0.

    """
    if ndim(tensors[0]) == 1:
        return torch.stack(tensors, 0)
    else:
        return torch.cat(tensors, 0)

def trange(start: int, end: int, step: int = 1) -> T.FloatTensor:
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
    return torch.range(start, end-1, step)

def euclidean_distance(a: T.FloatTensor, b: T.FloatTensor) -> float:
    """
    Compute the euclidean distance between two vectors.

    Args:
        a: A vector (i.e., 1D tensor).
        b: A vector (i.e., 1D tensor).

    Returns:
        float: Euclidean distance between a and b.

    """
    return (a - b).norm()

def squared_euclidean_distance(a: T.FloatTensor,
                               b: T.FloatTensor) -> float:
    """
    Compute the squared euclidean distance between two vectors.


    Args:
        a: A vector (i.e., 1D tensor).
        b: A vector (i.e., 1D tensor).

    Returns:
        float: Squared euclidean distance between a and b.

    """
    return euclidean_distance(a, b)**2

def resample(tensor: T.FloatTensor,
             n: int,
             replace: bool = True) -> T.FloatTensor:
    """
    Resample a tensor along the zeroth axis.

    Args:
        x: A tensor.
        n: Number of samples to take.
        replace (optional): Sample with replacement if True.

    Returns:
        tensor: Sample of size n from x along axis=0.

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

    Args:
        minibatch: A 2D tensor.
        samples: A 2D tensor.

    Returns:
        float: Approximate energy distance.

    Szekely, G.J. (2002)
    E-statistics: The Energy of Statistical Samples.
    Technical Report BGSU No 02-16.

    """
    d1 = 0
    d2 = 0
    d3 = 0

    n = min(len(minibatch), downsample)
    m = min(len(samples), downsample)

    X = resample(minibatch, n, replace=True)
    Y = resample(samples, m, replace=True)

    for i in range(n-1):
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

def replace_nan(x: T.FloatTensor, value: float = 0) -> T.FloatTensor:
    """
    Obtain a copy of the tensor with NaN replaced with a value,
    as well as the mask of elements for persisted values or NaN.

    Args:
        x: A tensor.
        value: The value to replace NaN with in tensor.

    Returns:
        t: a copy of the input tensor where NaN is replaced with a value.
        mask: a tensor denoting whether each element
              is a number (1/T) or NaN (0/F).

    """
    mask = (x==x)
    t = value * torch.ones(x.size())
    t.masked_copy_(mask, torch.masked_select(x, mask))
    return t, mask.float()
