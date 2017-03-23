backends
## class BroadcastError
BroadcastError exception:

Args: None


## functions

### acosh
```py

def acosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic cosine of a tensor.

Args:
    x (greater than 1): A tensor.

Returns:
    tensor: Elementwise inverse hyperbolic cosine.


### add\_dicts\_inplace
```py

def add_dicts_inplace(dict1: Dict[str, numpy.ndarray], dict2: Dict[str, numpy.ndarray]) -> Dict[str, numpy.ndarray]

```



Entrywise addition of dict2 to dict1.

Note:
    Modifies dict1 in place.

Args:
    dict1: A dictionary of tensors.
    dict2: A dictionary of tensors.

Returns:
    None


### affine
```py

def affine(a: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray

```



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


### allclose
```py

def allclose(x: numpy.ndarray, y: numpy.ndarray, rtol: float=1e-05, atol: float=1e-08) -> bool

```



Test if all elements in the two tensors are approximately equal.

absolute(x - y) <= (atol + rtol * absolute(y))

Args:
    x: A tensor.
    y: A tensor.
    rtol (optional): Relative tolerance.
    atol (optional): Absolute tolerance.

returns:
    bool: Check if all of the elements in the tensors are approximately equal.


### argmax
```py

def argmax(x: numpy.ndarray, axis: int) -> numpy.ndarray

```



Compute the indices of the maximal elements in x along the specified axis.

Args:
    x: A tensor:
    axis: The axis of interest.

Returns:
    tensor (of ints): Indices of the maximal elements in x along the
                      specified axis.


### argmin
```py

def argmin(x: numpy.ndarray, axis: int) -> numpy.ndarray

```



Compute the indices of the minimal elements in x along the specified axis.

Args:
    x: A tensor:
    axis: The axis of interest.

Returns:
    tensor (of ints): Indices of the minimum elements in x along the
                      specified axis.


### atanh
```py

def atanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic tangent of a tensor.

Args:
    x (between -1 and +1): A tensor.

Returns:
    tensor: Elementwise inverse hyperbolic tangent


### batch\_dot
```py

def batch_dot(vis: numpy.ndarray, W: numpy.ndarray, hid: numpy.ndarray, axis: int=1) -> numpy.ndarray

```



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


### batch\_outer
```py

def batch_outer(vis: numpy.ndarray, hid: numpy.ndarray) -> numpy.ndarray

```



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


### broadcast
```py

def broadcast(vec: numpy.ndarray, matrix: numpy.ndarray) -> numpy.ndarray

```



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


### clip
```py

def clip(tensor: numpy.ndarray, a_min: Union[int, float]=None, a_max: Union[int, float]=None) -> numpy.ndarray

```



Return a tensor with its values clipped between a_min and a_max.

Args:
    tensor: A tensor.
    a_min (optional): The desired lower bound on the elements of the tensor.
    a_max (optional): The desired upper bound on the elements of the tensor.

Returns:
    tensor: A new tensor with its values clipped between a_min and a_max.


### clip\_inplace
```py

def clip_inplace(tensor: numpy.ndarray, a_min: Union[int, float]=None, a_max: Union[int, float]=None) -> None

```



Clip the values of a tensor between a_min and a_max.

Note:
    Modifies tensor in place.

Args:
    tensor: A tensor.
    a_min (optional): The desired lower bound on the elements of the tensor.
    a_max (optional): The desired upper bound on the elements of the tensor.

Returns:
    None


### cos
```py

def cos(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise cosine of a tensor.

Args:
    x: A tensor.

Returns:
    tensor: Elementwise cosine.


### cosh
```py

def cosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic cosine of a tensor.

Args:
    x: A tensor.

Returns:
    tensor: Elementwise hyperbolic cosine.


### diag
```py

def diag(mat: numpy.ndarray) -> numpy.ndarray

```



Return the diagonal elements of a matrix.

Args:
    mat: A tensor.

Returns:
    tensor: A vector (i.e., 1D tensor) containing the diagonal
            elements of mat.


### diagonal\_matrix
```py

def diagonal_matrix(vec: numpy.ndarray) -> numpy.ndarray

```



Return a matrix with vec along the diagonal.

Args:
    vec: A vector (i.e., 1D tensor).

Returns:
    tensor: A matrix with the elements of vec along the diagonal,
            and zeros elsewhere.


### dot
```py

def dot(a: numpy.ndarray, b: numpy.ndarray) -> Union[numpy.float32, numpy.ndarray]

```



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


### dtype
```py

def dtype(tensor: numpy.ndarray) -> type

```



Return the type of the tensor.

Args:
    tensor: A tensor.

Returns:
    type: The type of the elements in the tensor.


### equal
```py

def equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise if two tensors are equal.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor (of bools): Elementwise test of equality between x and y.


### exp
```py

def exp(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise exponential function of a tensor.

Args:
    x: A tensor.

Returns:
    tensor (non-negative): Elementwise exponential.


### expit
```py

def expit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise expit (a.k.a. logistic) function of a tensor.

Args:
    x: A tensor.

Returns:
    tensor: Elementwise expit (a.k.a. logistic).


### fill\_diagonal
```py

def fill_diagonal(mat: numpy.ndarray, val: Union[int, float]) -> numpy.ndarray

```



Fill the diagonal of the matirx with a specified value.

Note:
    Modifies mat in place.

Args:
    mat: A tensor.
    val: The value to put along the diagonal.

Returns:
    None


### flatten
```py

def flatten(tensor: Union[numpy.float32, numpy.ndarray]) -> Union[numpy.float32, numpy.ndarray]

```



Return a flattened tensor.

Args:
    tensor: A tensor or scalar.

Returns:
    result: If arg is a tensor, return a flattened 1D tensor.
            If arg is a scalar, return the scalar.


### float\_scalar
```py

def float_scalar(scalar: Union[int, float]) -> float

```



Cast scalar to a 32-bit float.

Args:
    scalar: A scalar quantity:

Returns:
    numpy.float32: Scalar converted to floating point.


### float\_tensor
```py

def float_tensor(tensor: numpy.ndarray) -> numpy.ndarray

```



Cast tensor to a float tensor.

Args:
    tensor: A tensor.

Returns:
    tensor: Tensor converted to floating point.


### greater
```py

def greater(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x > y.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor (of bools): Elementwise test of x > y.


### greater\_equal
```py

def greater_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x >= y.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor (of bools): Elementwise test of x >= y.


### hstack
```py

def hstack(tensors: Iterable[numpy.ndarray]) -> numpy.ndarray

```



Concatenate tensors along the first axis.

Args:
    tensors: A list of tensors.

Returns:
    tensor: Tensors stacked along axis=1.


### identity
```py

def identity(n: int) -> numpy.ndarray

```



Return the n-dimensional identity matrix.

Args:
    n: The desired size of the tensor.

Returns:
    tensor: The n x n identity matrix with ones along the diagonal
            and zeros elsewhere.


### inv
```py

def inv(mat: numpy.ndarray) -> numpy.ndarray

```



Compute matrix inverse.

Args:
    mat: A square matrix.

Returns:
    tensor: The matrix inverse.


### jit
```py

def jit(signature_or_function=None, locals={}, target='cpu', cache=False, **options)

```



This decorator is used to compile a Python function into native code.

Args
-----
signature:
    The (optional) signature or list of signatures to be compiled.
    If not passed, required signatures will be compiled when the
    decorated function is called, depending on the argument values.
    As a convenience, you can directly pass the function to be compiled
    instead.

locals: dict
    Mapping of local variable names to Numba types. Used to override the
    types deduced by Numba's type inference engine.

target: str
    Specifies the target platform to compile for. Valid targets are cpu,
    gpu, npyufunc, and cuda. Defaults to cpu.

targetoptions:
    For a cpu target, valid options are:
        nopython: bool
            Set to True to disable the use of PyObjects and Python API
            calls. The default behavior is to allow the use of PyObjects
            and Python API. Default value is False.

        forceobj: bool
            Set to True to force the use of PyObjects for every value.
            Default value is False.

        looplift: bool
            Set to True to enable jitting loops in nopython mode while
            leaving surrounding code in object mode. This allows functions
            to allocate NumPy arrays and use Python objects, while the
            tight loops in the function can still be compiled in nopython
            mode. Any arrays that the tight loop uses should be created
            before the loop is entered. Default value is True.

Returns
--------
A callable usable as a compiled function.  Actual compiling will be
done lazily if no explicit signatures are passed.

Examples
--------
The function can be used in the following ways:

1) jit(signatures, target='cpu', **targetoptions) -> jit(function)

    Equivalent to:

        d = dispatcher(function, targetoptions)
        for signature in signatures:
            d.compile(signature)

    Create a dispatcher object for a python function.  Then, compile
    the function with the given signature(s).

    Example:

        @jit("int32(int32, int32)")
        def foo(x, y):
            return x + y

        @jit(["int32(int32, int32)", "float32(float32, float32)"])
        def bar(x, y):
            return x + y

2) jit(function, target='cpu', **targetoptions) -> dispatcher

    Create a dispatcher function object that specializes at call site.

    Examples:

        @jit
        def foo(x, y):
            return x + y

        @jit(target='cpu', nopython=True)
        def bar(x, y):
            return x + y


### lesser
```py

def lesser(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x < y.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor (of bools): Elementwise test of x < y.


### lesser\_equal
```py

def lesser_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x <= y.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor (of bools): Elementwise test of x <= y.


### log
```py

def log(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise natural logarithm of a tensor.

Args:
    x (non-negative): A tensor.

Returns:
    tensor: Elementwise natural logarithm.


### logaddexp
```py

def logaddexp(x1: numpy.ndarray, x2: numpy.ndarray) -> numpy.ndarray

```



Elementwise logaddexp function: log(exp(x1) + exp(x2))

Args:
    x1: A tensor.
    x2: A tensor.

Returns:
    tensor: Elementwise logaddexp.


### logcosh
```py

def logcosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logarithm of the hyperbolic cosine of a tensor.

Args:
    x: A tensor.

Returns:
    tensor: Elementwise logarithm of the hyperbolic cosine.


### logit
```py

def logit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logit function of a tensor. Inverse of the expit function.

Args:
    x (between 0 and 1): A tensor.

Returns:
    tensor: Elementwise logit function


### maximum
```py

def maximum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise maximum of two tensors.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor: Elementwise maximum of x and y.


### mean
```py

def mean(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



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


### minimum
```py

def minimum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise minimum of two tensors.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor: Elementwise minimum of x and y.


### mix\_inplace
```py

def mix_inplace(w: Union[int, float], x: numpy.ndarray, y: numpy.ndarray) -> None

```



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


### multiply\_dict\_inplace
```py

def multiply_dict_inplace(dict1: Dict[str, numpy.ndarray], scalar: Union[int, float]) -> None

```



Entrywise multiplication of dict1 by scalar.

Note:
    Modifies dict1 in place.

Args:
    dict1: A dictionary of tensors.
    scalar: A scalar.

Returns:
    None


### ndim
```py

def ndim(tensor: numpy.ndarray) -> int

```



Return the number of dimensions of a tensor.

Args:
    tensor: A tensor:

Returns:
    int: The number of dimensions of the tensor.


### norm
```py

def norm(x: numpy.ndarray) -> float

```



Return the L2 norm of a tensor.

Args:
    x: A tensor.

Returns:
    float: The L2 norm of the tensor
           (i.e., the sqrt of the sum of the squared elements).


### normalize
```py

def normalize(x: numpy.ndarray) -> numpy.ndarray

```



Divide x by it's sum.

Args:
    x: A non-negative tensor.

Returns:
    tensor: A tensor normalized by it's sum.


### not\_equal
```py

def not_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if two tensors are not equal.

Args:
    x: A tensor.
    y: A tensor.

Returns:
    tensor (of bools): Elementwise test of non-equality between x and y.


### ones
```py

def ones(shape: Tuple[int]) -> numpy.ndarray

```



Return a tensor of a specified shape filled with ones.

Args:
    shape: The shape of the desired tensor.

Returns:
    tensor: A tensor of ones with the desired shape.


### ones\_like
```py

def ones_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor of ones with the same shape as the input tensor.

Args:
    tensor: A tensor.

Returns:
    tensor: A tensor with the same shape.


### outer
```py

def outer(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the outer product of vectors x and y.

mat_{ij} = x_i * y_j

Args:
    x: A vector (i.e., a 1D tensor).
    y: A vector (i.e., a 1D tensor).

Returns:
    tensor: Outer product of vectors x and y.


### quadratic
```py

def quadratic(a: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray

```



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


### rand
```py

def rand(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with uniform random numbers
between 0 and 1.

Args:
    shape: Desired shape of the random tensor.

Returns:
    tensor: Random numbers between 0 and 1.


### randn
```py

def randn(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with random numbers
drawn from a standard normal distribution (mean = 0, variance = 1).

Args:
    shape: Desired shape of the random tensor.

Returns:
    tensor: Random numbers between from a standard normal distribution.


### reciprocal
```py

def reciprocal(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse of a tensor.

Args:
    x (non-zero): A tensor:

Returns:
    tensor: Elementwise inverse.


### repeat
```py

def repeat(tensor: numpy.ndarray, n: int) -> numpy.ndarray

```



Repeat tensor n times along the first axis.

Args:
    tensor: A vector (i.e., 1D tensor).
    n: The number of repeats.

Returns:
    tensor: A vector created from many repeats of the input tensor.


### reshape
```py

def reshape(tensor: numpy.ndarray, newshape: Tuple[int]) -> numpy.ndarray

```



Return tensor with a new shape.

Args:
    tensor: A tensor.
    newshape: The desired shape.

Returns:
    tensor: A tensor with the desired shape.


### set\_seed
```py

def set_seed(n: int=137) -> None

```



Set the seed of the random number generator.

Notes:
    Default seed is 137.

Args:
    n: Random seed.

Returns:
    None


### shape
```py

def shape(tensor: numpy.ndarray) -> Tuple[int]

```



Return a tuple with the shape of the tensor.

Args:
    tensor: A tensor:

Returns:
    tuple: A tuple of integers describing the shape of the tensor.


### sign
```py

def sign(tensor: numpy.ndarray) -> numpy.ndarray

```



Return the elementwise sign of a tensor.

Args:
    tensor: A tensor.

Returns:
    tensor: The sign of the elements in the tensor.


### sin
```py

def sin(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise sine of a tensor.

Args:
    x: A tensor.

Returns:
    tensor: Elementwise sine.


### softplus
```py

def softplus(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise softplus function of a tensor.

Args:
    x: A tensor.

Returns:
    tensor: Elementwise softplus.


### sqrt
```py

def sqrt(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square root of a tensor.

Args:
    x (non-negative): A tensor.

Returns:
    tensor(non-negative): Elementwise square root.


### sqrt\_div
```py

def sqrt_div(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise division of x by sqrt(y).

Args:
    x: A tensor:
    y: A non-negative tensor.

Returns:
    tensor: Elementwise division of x by sqrt(y).


### square
```py

def square(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square of a tensor.

Args:
    x: A tensor.

Returns:
    tensor (non-negative): Elementwise square.


### square\_mix\_inplace
```py

def square_mix_inplace(w: Union[int, float], x: numpy.ndarray, y: numpy.ndarray) -> None

```



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


### stack
```py

def stack(tensors: Iterable[numpy.ndarray], axis: int) -> numpy.ndarray

```



Stack tensors along the specified axis.

Args:
    tensors: A list of tensors.
    axis: The axis the tensors will be stacked along.

Returns:
    tensor: Stacked tensors from the input list.


### std
```py

def std(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



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


### subtract\_dicts\_inplace
```py

def subtract_dicts_inplace(dict1: Dict[str, numpy.ndarray], dict2: Dict[str, numpy.ndarray]) -> Dict[str, numpy.ndarray]

```



Entrywise subtraction of dict2 from dict1.

Note:
    Modifies dict1 in place.

Args:
    dict1: A dictionary of tensors.
    dict2: A dictionary of tensors.

Returns:
    None


### tabs
```py

def tabs(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise absolute value of a tensor.

Args:
    x: A tensor.

Returns:
    tensor (non-negative): Absolute value of x.


### tall
```py

def tall(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[bool, numpy.ndarray]

```



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


### tanh
```py

def tanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic tangent of a tensor.

Args:
    x: A tensor.

Returns:
    tensor: Elementwise hyperbolic tangent.


### tany
```py

def tany(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[bool, numpy.ndarray]

```



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


### tmax
```py

def tmax(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



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


### tmin
```py

def tmin(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



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


### to\_numpy\_array
```py

def to_numpy_array(tensor: numpy.ndarray) -> numpy.ndarray

```



Return tensor as a numpy array.

Args:
    tensor: A tensor.

Returns:
    tensor: Tensor converted to a numpy array.


### tpow
```py

def tpow(x: numpy.ndarray, a: float) -> numpy.ndarray

```



Elementwise power of a tensor x to power a.

Args:
    x: A tensor.
    a: Power.

Returns:
    tensor: Elementwise x to the power of a.


### tprod
```py

def tprod(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



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


### trange
```py

def trange(start: int, end: int, step: int=1) -> numpy.ndarray

```



Generate a tensor like a python range.

Args:
    start: The start of the range.
    end: The end of the range.
    step: The step of the range.

Returns:
    tensor: A vector ranging from start to end in increments
            of step. Cast to float rather than int.


### transpose
```py

def transpose(tensor: numpy.ndarray) -> numpy.ndarray

```



Return the transpose of a tensor.

Args:
    tensor: A tensor.

Returns:
    tensor: The transpose (exchange of rows and columns) of the tensor.


### tround
```py

def tround(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor with rounded elements.

Args:
    tensor: A tensor.

Returns:
    tensor: A tensor rounded to the nearest integer (still floating point).


### tsum
```py

def tsum(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



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


### var
```py

def var(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



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


### vstack
```py

def vstack(tensors: Iterable[numpy.ndarray]) -> numpy.ndarray

```



Concatenate tensors along the zeroth axis.

Args:
    tensors: A list of tensors.

Returns:
    tensor: Tensors stacked along axis=0.


### zeros
```py

def zeros(shape: Tuple[int]) -> numpy.ndarray

```



Return a tensor of a specified shape filled with zeros.

Args:
    shape: The shape of the desired tensor.

Returns:
    tensor: A tensor of zeros with the desired shape.


### zeros\_like
```py

def zeros_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor of zeros with the same shape as the input tensor.

Args:
    tensor: A tensor.

Returns:
    tensor: A tensor of zeros with the same shape.

