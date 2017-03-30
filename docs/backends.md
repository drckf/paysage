# Documentation for Backends (backends.py)

## class BroadcastError
BroadcastError exception:<br /><br />Args: None


## functions

### acosh
```py

def acosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic cosine of a tensor.<br /><br />Args:<br /> ~ x (greater than 1): A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise inverse hyperbolic cosine.


### add
```py

def add(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Add tensor a to tensor b using broadcasting.<br /><br />Args:<br /> ~ a: A tensor<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ tensor: a + b


### add\_dicts\_inplace
```py

def add_dicts_inplace(dict1: Dict[str, numpy.ndarray], dict2: Dict[str, numpy.ndarray]) -> Dict[str, numpy.ndarray]

```



Entrywise addition of dict2 to dict1.<br /><br />Note:<br /> ~ Modifies dict1 in place.<br /><br />Args:<br /> ~ dict1: A dictionary of tensors.<br /> ~ dict2: A dictionary of tensors.<br /><br />Returns:<br /> ~ None


### add\_tuples
```py

def add_tuples(a, b)

```



Add tuple a to tuple b entrywise.<br /><br />Args:<br /> ~ a (namedtuple): (key: tensor)<br /> ~ b (namedtuple): (key: tensor)<br /><br />Returns:<br /> ~ namedtuple: a + b (key: tensor)


### affine
```py

def affine(a: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray

```



Evaluate the affine transformation a + W b.<br /><br />a ~ vector, b ~ vector, W ~ matrix:<br />a_i + \sum_j W_ij b_j<br /><br />a ~ matrix, b ~ matrix, W ~ matrix:<br />a_ij + \sum_k W_ik b_kj<br /><br />Args:<br /> ~ a: A tensor (1 or 2 dimensional).<br /> ~ b: A tensor (1 or 2 dimensional).<br /> ~ W: A tensor (2 dimensional).<br /><br />Returns:<br /> ~ tensor: Affine transformation a + W b.


### allclose
```py

def allclose(x: numpy.ndarray, y: numpy.ndarray, rtol: float=1e-05, atol: float=1e-08) -> bool

```



Test if all elements in the two tensors are approximately equal.<br /><br />absolute(x - y) <= (atol + rtol * absolute(y))<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /> ~ rtol (optional): Relative tolerance.<br /> ~ atol (optional): Absolute tolerance.<br /><br />returns:<br /> ~ bool: Check if all of the elements in the tensors are approximately equal.


### argmax
```py

def argmax(x: numpy.ndarray, axis: int) -> numpy.ndarray

```



Compute the indices of the maximal elements in x along the specified axis.<br /><br />Args:<br /> ~ x: A tensor:<br /> ~ axis: The axis of interest.<br /><br />Returns:<br /> ~ tensor (of ints): Indices of the maximal elements in x along the<br /> ~ specified axis.


### argmin
```py

def argmin(x: numpy.ndarray, axis: int) -> numpy.ndarray

```



Compute the indices of the minimal elements in x along the specified axis.<br /><br />Args:<br /> ~ x: A tensor:<br /> ~ axis: The axis of interest.<br /><br />Returns:<br /> ~ tensor (of ints): Indices of the minimum elements in x along the<br /> ~ specified axis.


### atanh
```py

def atanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic tangent of a tensor.<br /><br />Args:<br /> ~ x (between -1 and +1): A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise inverse hyperbolic tangent


### batch\_dot
```py

def batch_dot(vis: numpy.ndarray, W: numpy.ndarray, hid: numpy.ndarray, axis: int=1) -> numpy.ndarray

```



Let v by a L x N matrix where each row v_i is a visible vector.<br />Let h be a L x M matrix where each row h_i is a hidden vector.<br />And, let W be a N x M matrix of weights.<br />Then, batch_dot(v,W,h) = \sum_i v_i^T W h_i<br /><br />The actual computation is performed with a vectorized expression.<br /><br />Args:<br /> ~ vis: A tensor.<br /> ~ W: A tensor.<br /> ~ hid: A tensor.<br /> ~ axis (optional): Axis of interest<br /><br />Returns:<br /> ~ tensor: A vector.


### batch\_outer
```py

def batch_outer(vis: numpy.ndarray, hid: numpy.ndarray) -> numpy.ndarray

```



Let v by a L x N matrix where each row v_i is a visible vector.<br />Let h be a L x M matrix where each row h_i is a hidden vector.<br />Then, batch_outer(v, h) = \sum_i v_i h_i^T<br />Returns an N x M matrix.<br /><br />The actual computation is performed with a vectorized expression.<br /><br />Args:<br /> ~ vis: A tensor.<br /> ~ hid: A tensor:<br /><br />Returns:<br /> ~ tensor: A matrix.


### broadcast
```py

def broadcast(vec: numpy.ndarray, matrix: numpy.ndarray) -> numpy.ndarray

```



Broadcasts vec into the shape of matrix following numpy rules:<br /><br />vec ~ (N, 1) broadcasts to matrix ~ (N, M)<br />vec ~ (1, N) and (N,) broadcast to matrix ~ (M, N)<br /><br />Args:<br /> ~ vec: A vector (either flat, row, or column).<br /> ~ matrix: A matrix (i.e., a 2D tensor).<br /><br />Returns:<br /> ~ tensor: A tensor of the same size as matrix containing the elements<br /> ~  ~  ~ of the vector.<br /><br />Raises:<br /> ~ BroadcastError


### clip
```py

def clip(tensor: numpy.ndarray, a_min: Union[int, float]=None, a_max: Union[int, float]=None) -> numpy.ndarray

```



Return a tensor with its values clipped between a_min and a_max.<br /><br />Args:<br /> ~ tensor: A tensor.<br /> ~ a_min (optional): The desired lower bound on the elements of the tensor.<br /> ~ a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br /> ~ tensor: A new tensor with its values clipped between a_min and a_max.


### clip\_inplace
```py

def clip_inplace(tensor: numpy.ndarray, a_min: Union[int, float]=None, a_max: Union[int, float]=None) -> None

```



Clip the values of a tensor between a_min and a_max.<br /><br />Note:<br /> ~ Modifies tensor in place.<br /><br />Args:<br /> ~ tensor: A tensor.<br /> ~ a_min (optional): The desired lower bound on the elements of the tensor.<br /> ~ a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br /> ~ None


### cos
```py

def cos(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise cosine of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise cosine.


### cosh
```py

def cosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic cosine of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise hyperbolic cosine.


### diag
```py

def diag(mat: numpy.ndarray) -> numpy.ndarray

```



Return the diagonal elements of a matrix.<br /><br />Args:<br /> ~ mat: A tensor.<br /><br />Returns:<br /> ~ tensor: A vector (i.e., 1D tensor) containing the diagonal<br /> ~  ~  ~ elements of mat.


### diagonal\_matrix
```py

def diagonal_matrix(vec: numpy.ndarray) -> numpy.ndarray

```



Return a matrix with vec along the diagonal.<br /><br />Args:<br /> ~ vec: A vector (i.e., 1D tensor).<br /><br />Returns:<br /> ~ tensor: A matrix with the elements of vec along the diagonal,<br /> ~  ~  ~ and zeros elsewhere.


### divide
```py

def divide(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Divide tensor b by tensor a using broadcasting.<br /><br />Args:<br /> ~ a: A tensor (non-zero)<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ tensor: a * b


### divide\_tuples
```py

def divide_tuples(a, b)

```



Divide tuple b by tuple a entrywise.<br /><br />Args:<br /> ~ a (namedtuple; non-zero): (key: tensor)<br /> ~ b (namedtuple): (key: tensor)<br /><br />Returns:<br /> ~ namedtuple: b / a (key: tensor)


### dot
```py

def dot(a: numpy.ndarray, b: numpy.ndarray) -> Union[numpy.float32, numpy.ndarray]

```



Compute the matrix/dot product of tensors a and b.<br /><br />Vector-Vector:<br /> ~ \sum_i a_i b_i<br /><br />Matrix-Vector:<br /> ~ \sum_j a_ij b_j<br /><br />Matrix-Matrix:<br /> ~ \sum_j a_ij b_jk<br /><br />Args:<br /> ~ a: A tensor.<br /> ~ b: A tensor:<br /><br />Returns:<br /> ~ if a and b are 1-dimensions:<br /> ~  ~ float: the dot product of vectors a and b<br /> ~ else:<br /> ~  ~ tensor: the matrix product of tensors a and b


### dtype
```py

def dtype(tensor: numpy.ndarray) -> type

```



Return the type of the tensor.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ type: The type of the elements in the tensor.


### equal
```py

def equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise if two tensors are equal.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor (of bools): Elementwise test of equality between x and y.


### exp
```py

def exp(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise exponential function of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor (non-negative): Elementwise exponential.


### expit
```py

def expit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise expit (a.k.a. logistic) function of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise expit (a.k.a. logistic).


### fill\_diagonal
```py

def fill_diagonal(mat: numpy.ndarray, val: Union[int, float]) -> numpy.ndarray

```



Fill the diagonal of the matirx with a specified value.<br /><br />Note:<br /> ~ Modifies mat in place.<br /><br />Args:<br /> ~ mat: A tensor.<br /> ~ val: The value to put along the diagonal.<br /><br />Returns:<br /> ~ None


### flatten
```py

def flatten(tensor: Union[numpy.float32, numpy.ndarray]) -> Union[numpy.float32, numpy.ndarray]

```



Return a flattened tensor.<br /><br />Args:<br /> ~ tensor: A tensor or scalar.<br /><br />Returns:<br /> ~ result: If arg is a tensor, return a flattened 1D tensor.<br /> ~  ~  ~ If arg is a scalar, return the scalar.


### float\_scalar
```py

def float_scalar(scalar: Union[int, float]) -> float

```



Cast scalar to a 32-bit float.<br /><br />Args:<br /> ~ scalar: A scalar quantity:<br /><br />Returns:<br /> ~ numpy.float32: Scalar converted to floating point.


### float\_tensor
```py

def float_tensor(tensor: numpy.ndarray) -> numpy.ndarray

```



Cast tensor to a float tensor.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: Tensor converted to floating point.


### greater
```py

def greater(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x > y.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor (of bools): Elementwise test of x > y.


### greater\_equal
```py

def greater_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x >= y.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor (of bools): Elementwise test of x >= y.


### hstack
```py

def hstack(tensors: Iterable[numpy.ndarray]) -> numpy.ndarray

```



Concatenate tensors along the first axis.<br /><br />Args:<br /> ~ tensors: A list of tensors.<br /><br />Returns:<br /> ~ tensor: Tensors stacked along axis=1.


### identity
```py

def identity(n: int) -> numpy.ndarray

```



Return the n-dimensional identity matrix.<br /><br />Args:<br /> ~ n: The desired size of the tensor.<br /><br />Returns:<br /> ~ tensor: The n x n identity matrix with ones along the diagonal<br /> ~  ~  ~ and zeros elsewhere.


### inv
```py

def inv(mat: numpy.ndarray) -> numpy.ndarray

```



Compute matrix inverse.<br /><br />Args:<br /> ~ mat: A square matrix.<br /><br />Returns:<br /> ~ tensor: The matrix inverse.


### jit
```py

def jit(signature_or_function=None, locals={}, target='cpu', cache=False, **options)

```



This decorator is used to compile a Python function into native code.<br /><br />Args<br />-----<br />signature:<br /> ~ The (optional) signature or list of signatures to be compiled.<br /> ~ If not passed, required signatures will be compiled when the<br /> ~ decorated function is called, depending on the argument values.<br /> ~ As a convenience, you can directly pass the function to be compiled<br /> ~ instead.<br /><br />locals: dict<br /> ~ Mapping of local variable names to Numba types. Used to override the<br /> ~ types deduced by Numba's type inference engine.<br /><br />target: str<br /> ~ Specifies the target platform to compile for. Valid targets are cpu,<br /> ~ gpu, npyufunc, and cuda. Defaults to cpu.<br /><br />targetoptions:<br /> ~ For a cpu target, valid options are:<br /> ~  ~ nopython: bool<br /> ~  ~  ~ Set to True to disable the use of PyObjects and Python API<br /> ~  ~  ~ calls. The default behavior is to allow the use of PyObjects<br /> ~  ~  ~ and Python API. Default value is False.<br /><br /> ~  ~ forceobj: bool<br /> ~  ~  ~ Set to True to force the use of PyObjects for every value.<br /> ~  ~  ~ Default value is False.<br /><br /> ~  ~ looplift: bool<br /> ~  ~  ~ Set to True to enable jitting loops in nopython mode while<br /> ~  ~  ~ leaving surrounding code in object mode. This allows functions<br /> ~  ~  ~ to allocate NumPy arrays and use Python objects, while the<br /> ~  ~  ~ tight loops in the function can still be compiled in nopython<br /> ~  ~  ~ mode. Any arrays that the tight loop uses should be created<br /> ~  ~  ~ before the loop is entered. Default value is True.<br /><br />Returns<br />--------<br />A callable usable as a compiled function.  Actual compiling will be<br />done lazily if no explicit signatures are passed.<br /><br />Examples<br />--------<br />The function can be used in the following ways:<br /><br />1) jit(signatures, target='cpu', **targetoptions) -> jit(function)<br /><br /> ~ Equivalent to:<br /><br /> ~  ~ d = dispatcher(function, targetoptions)<br /> ~  ~ for signature in signatures:<br /> ~  ~  ~ d.compile(signature)<br /><br /> ~ Create a dispatcher object for a python function.  Then, compile<br /> ~ the function with the given signature(s).<br /><br /> ~ Example:<br /><br /> ~  ~ @jit("int32(int32, int32)")<br /> ~  ~ def foo(x, y):<br /> ~  ~  ~ return x + y<br /><br /> ~  ~ @jit(["int32(int32, int32)", "float32(float32, float32)"])<br /> ~  ~ def bar(x, y):<br /> ~  ~  ~ return x + y<br /><br />2) jit(function, target='cpu', **targetoptions) -> dispatcher<br /><br /> ~ Create a dispatcher function object that specializes at call site.<br /><br /> ~ Examples:<br /><br /> ~  ~ @jit<br /> ~  ~ def foo(x, y):<br /> ~  ~  ~ return x + y<br /><br /> ~  ~ @jit(target='cpu', nopython=True)<br /> ~  ~ def bar(x, y):<br /> ~  ~  ~ return x + y


### lesser
```py

def lesser(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x < y.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor (of bools): Elementwise test of x < y.


### lesser\_equal
```py

def lesser_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x <= y.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor (of bools): Elementwise test of x <= y.


### log
```py

def log(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise natural logarithm of a tensor.<br /><br />Args:<br /> ~ x (non-negative): A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise natural logarithm.


### logaddexp
```py

def logaddexp(x1: numpy.ndarray, x2: numpy.ndarray) -> numpy.ndarray

```



Elementwise logaddexp function: log(exp(x1) + exp(x2))<br /><br />Args:<br /> ~ x1: A tensor.<br /> ~ x2: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise logaddexp.


### logcosh
```py

def logcosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logarithm of the hyperbolic cosine of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise logarithm of the hyperbolic cosine.


### logit
```py

def logit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logit function of a tensor. Inverse of the expit function.<br /><br />Args:<br /> ~ x (between 0 and 1): A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise logit function


### maximum
```py

def maximum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise maximum of two tensors.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise maximum of x and y.


### mean
```py

def mean(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the mean of the elements of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis for taking the mean.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall mean of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The mean of the tensor along the specified axis.


### minimum
```py

def minimum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise minimum of two tensors.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise minimum of x and y.


### mix\_inplace
```py

def mix_inplace(w: Union[int, float], x: numpy.ndarray, y: numpy.ndarray) -> None

```



Compute a weighted average of two matrices (x and y) and store the results in x.<br />Useful for keeping track of running averages during training.<br /><br />x <- w * x + (1-w) * y<br /><br />Note:<br /> ~ Modifies x in place.<br /><br />Args:<br /> ~ w: The mixing coefficient between 0 and 1 .<br /> ~ x: A tensor.<br /> ~ y: A tensor:<br /><br />Returns:<br /> ~ None


### multiply
```py

def multiply(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Multiply tensor b with tensor a using broadcasting.<br /><br />Args:<br /> ~ a: A tensor<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ tensor: a * b


### multiply\_dict\_inplace
```py

def multiply_dict_inplace(dict1: Dict[str, numpy.ndarray], scalar: Union[int, float]) -> None

```



Entrywise multiplication of dict1 by scalar.<br /><br />Note:<br /> ~ Modifies dict1 in place.<br /><br />Args:<br /> ~ dict1: A dictionary of tensors.<br /> ~ scalar: A scalar.<br /><br />Returns:<br /> ~ None


### multiply\_tuples
```py

def multiply_tuples(a, b)

```



Multiply tuple b by tuple a entrywise.<br /><br />Args:<br /> ~ a (namedtuple): (key: tensor)<br /> ~ b (namedtuple): (key: tensor)<br /><br />Returns:<br /> ~ namedtuple: a * b (key: tensor)


### namedtuple
```py

def namedtuple(typename, field_names, verbose=False, rename=False)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)


### ndim
```py

def ndim(tensor: numpy.ndarray) -> int

```



Return the number of dimensions of a tensor.<br /><br />Args:<br /> ~ tensor: A tensor:<br /><br />Returns:<br /> ~ int: The number of dimensions of the tensor.


### norm
```py

def norm(x: numpy.ndarray) -> float

```



Return the L2 norm of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ float: The L2 norm of the tensor<br /> ~  ~    (i.e., the sqrt of the sum of the squared elements).


### normalize
```py

def normalize(x: numpy.ndarray) -> numpy.ndarray

```



Divide x by it's sum.<br /><br />Args:<br /> ~ x: A non-negative tensor.<br /><br />Returns:<br /> ~ tensor: A tensor normalized by it's sum.


### not\_equal
```py

def not_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if two tensors are not equal.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor (of bools): Elementwise test of non-equality between x and y.


### ones
```py

def ones(shape: Tuple[int]) -> numpy.ndarray

```



Return a tensor of a specified shape filled with ones.<br /><br />Args:<br /> ~ shape: The shape of the desired tensor.<br /><br />Returns:<br /> ~ tensor: A tensor of ones with the desired shape.


### ones\_like
```py

def ones_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor of ones with the same shape as the input tensor.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: A tensor with the same shape.


### outer
```py

def outer(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the outer product of vectors x and y.<br /><br />mat_{ij} = x_i * y_j<br /><br />Args:<br /> ~ x: A vector (i.e., a 1D tensor).<br /> ~ y: A vector (i.e., a 1D tensor).<br /><br />Returns:<br /> ~ tensor: Outer product of vectors x and y.


### quadratic
```py

def quadratic(a: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray

```



Evaluate the quadratic form a W b.<br /><br />a ~ vector, b ~ vector, W ~ matrix:<br />\sum_ij a_i W_ij b_j<br /><br />a ~ matrix, b ~ matrix, W ~ matrix:<br />\sum_kl a_ik W_kl b_lj<br /><br />Args:<br /> ~ a: A tensor:<br /> ~ b: A tensor:<br /> ~ W: A tensor:<br /><br />Returns:<br /> ~ tensor: Quadratic function a W b.


### rand
```py

def rand(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with uniform random numbers<br />between 0 and 1.<br /><br />Args:<br /> ~ shape: Desired shape of the random tensor.<br /><br />Returns:<br /> ~ tensor: Random numbers between 0 and 1.


### randn
```py

def randn(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with random numbers<br />drawn from a standard normal distribution (mean = 0, variance = 1).<br /><br />Args:<br /> ~ shape: Desired shape of the random tensor.<br /><br />Returns:<br /> ~ tensor: Random numbers between from a standard normal distribution.


### reciprocal
```py

def reciprocal(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse of a tensor.<br /><br />Args:<br /> ~ x (non-zero): A tensor:<br /><br />Returns:<br /> ~ tensor: Elementwise inverse.


### repeat
```py

def repeat(tensor: numpy.ndarray, n: int) -> numpy.ndarray

```



Repeat tensor n times along the first axis.<br /><br />Args:<br /> ~ tensor: A vector (i.e., 1D tensor).<br /> ~ n: The number of repeats.<br /><br />Returns:<br /> ~ tensor: A vector created from many repeats of the input tensor.


### reshape
```py

def reshape(tensor: numpy.ndarray, newshape: Tuple[int]) -> numpy.ndarray

```



Return tensor with a new shape.<br /><br />Args:<br /> ~ tensor: A tensor.<br /> ~ newshape: The desired shape.<br /><br />Returns:<br /> ~ tensor: A tensor with the desired shape.


### set\_seed
```py

def set_seed(n: int=137) -> None

```



Set the seed of the random number generator.<br /><br />Notes:<br /> ~ Default seed is 137.<br /><br />Args:<br /> ~ n: Random seed.<br /><br />Returns:<br /> ~ None


### shape
```py

def shape(tensor: numpy.ndarray) -> Tuple[int]

```



Return a tuple with the shape of the tensor.<br /><br />Args:<br /> ~ tensor: A tensor:<br /><br />Returns:<br /> ~ tuple: A tuple of integers describing the shape of the tensor.


### sign
```py

def sign(tensor: numpy.ndarray) -> numpy.ndarray

```



Return the elementwise sign of a tensor.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: The sign of the elements in the tensor.


### sin
```py

def sin(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise sine of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise sine.


### softplus
```py

def softplus(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise softplus function of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise softplus.


### sqrt
```py

def sqrt(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square root of a tensor.<br /><br />Args:<br /> ~ x (non-negative): A tensor.<br /><br />Returns:<br /> ~ tensor(non-negative): Elementwise square root.


### sqrt\_div
```py

def sqrt_div(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise division of x by sqrt(y).<br /><br />Args:<br /> ~ x: A tensor:<br /> ~ y: A non-negative tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise division of x by sqrt(y).


### square
```py

def square(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor (non-negative): Elementwise square.


### square\_mix\_inplace
```py

def square_mix_inplace(w: Union[int, float], x: numpy.ndarray, y: numpy.ndarray) -> None

```



Compute a weighted average of two matrices (x and y^2) and store the results in x.<br />Useful for keeping track of running averages of squared matrices during training.<br /><br />x <- w x + (1-w) * y**2<br /><br />Note:<br /> ~ Modifies x in place.<br /><br />Args:<br /> ~ w: The mixing coefficient between 0 and 1 .<br /> ~ x: A tensor.<br /> ~ y: A tensor:<br /><br />Returns:<br /> ~ None


### stack
```py

def stack(tensors: Iterable[numpy.ndarray], axis: int) -> numpy.ndarray

```



Stack tensors along the specified axis.<br /><br />Args:<br /> ~ tensors: A list of tensors.<br /> ~ axis: The axis the tensors will be stacked along.<br /><br />Returns:<br /> ~ tensor: Stacked tensors from the input list.


### std
```py

def std(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the standard deviation of the elements of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis for taking the standard deviation.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall standard deviation of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The standard deviation of the tensor along the specified axis.


### subtract
```py

def subtract(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Subtract tensor a from tensor b using broadcasting.<br /><br />Args:<br /> ~ a: A tensor<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ tensor: b - a


### subtract\_dicts\_inplace
```py

def subtract_dicts_inplace(dict1: Dict[str, numpy.ndarray], dict2: Dict[str, numpy.ndarray]) -> Dict[str, numpy.ndarray]

```



Entrywise subtraction of dict2 from dict1.<br /><br />Note:<br /> ~ Modifies dict1 in place.<br /><br />Args:<br /> ~ dict1: A dictionary of tensors.<br /> ~ dict2: A dictionary of tensors.<br /><br />Returns:<br /> ~ None


### subtract\_tuples
```py

def subtract_tuples(a, b)

```



Subtract tuple a from tuple b entrywise.<br /><br />Args:<br /> ~ a (namedtuple): (key: tensor)<br /> ~ b (namedtuple): (key: tensor)<br /><br />Returns:<br /> ~ namedtuple: b - a (key: tensor)


### tabs
```py

def tabs(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise absolute value of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor (non-negative): Absolute value of x.


### tall
```py

def tall(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[bool, numpy.ndarray]

```



Return True if all elements of the input tensor are true along the<br />specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis of interest.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ bool: 'all' applied to all elements in the tensor<br /> ~ else:<br /> ~  ~ tensor (of bools): 'all' applied to the elements in the tensor<br /> ~  ~  ~  ~  ~  ~  ~ along axis


### tanh
```py

def tanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic tangent of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise hyperbolic tangent.


### tany
```py

def tany(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[bool, numpy.ndarray]

```



Return True if any elements of the input tensor are true along the<br />specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis of interest.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ bool: 'any' applied to all elements in the tensor<br /> ~ else:<br /> ~  ~ tensor (of bools): 'any' applied to the elements in the tensor<br /> ~  ~  ~  ~  ~  ~  ~ along axis


### tmax
```py

def tmax(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the elementwise maximum of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis for taking the maximum.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall maximum of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The maximum of the tensor along the specified axis.


### tmin
```py

def tmin(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the elementwise minimum of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis for taking the minimum.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall minimum of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The minimum of the tensor along the specified axis.


### to\_numpy\_array
```py

def to_numpy_array(tensor: numpy.ndarray) -> numpy.ndarray

```



Return tensor as a numpy array.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: Tensor converted to a numpy array.


### tpow
```py

def tpow(x: numpy.ndarray, a: float) -> numpy.ndarray

```



Elementwise power of a tensor x to power a.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ a: Power.<br /><br />Returns:<br /> ~ tensor: Elementwise x to the power of a.


### tprod
```py

def tprod(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the product of the elements of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis for taking the product.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall product of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The product of the tensor along the specified axis.


### trange
```py

def trange(start: int, end: int, step: int=1) -> numpy.ndarray

```



Generate a tensor like a python range.<br /><br />Args:<br /> ~ start: The start of the range.<br /> ~ end: The end of the range.<br /> ~ step: The step of the range.<br /><br />Returns:<br /> ~ tensor: A vector ranging from start to end in increments<br /> ~  ~  ~ of step. Cast to float rather than int.


### transpose
```py

def transpose(tensor: numpy.ndarray) -> numpy.ndarray

```



Return the transpose of a tensor.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: The transpose (exchange of rows and columns) of the tensor.


### tround
```py

def tround(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor with rounded elements.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: A tensor rounded to the nearest integer (still floating point).


### tsum
```py

def tsum(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the sum of the elements of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis for taking the sum.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall sum of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The sum of the tensor along the specified axis.


### var
```py

def var(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the variance of the elements of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor.<br /> ~ axis (optional): The axis for taking the variance.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall variance of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The variance of the tensor along the specified axis.


### vstack
```py

def vstack(tensors: Iterable[numpy.ndarray]) -> numpy.ndarray

```



Concatenate tensors along the zeroth axis.<br /><br />Args:<br /> ~ tensors: A list of tensors.<br /><br />Returns:<br /> ~ tensor: Tensors stacked along axis=0.


### zeros
```py

def zeros(shape: Tuple[int]) -> numpy.ndarray

```



Return a tensor of a specified shape filled with zeros.<br /><br />Args:<br /> ~ shape: The shape of the desired tensor.<br /><br />Returns:<br /> ~ tensor: A tensor of zeros with the desired shape.


### zeros\_like
```py

def zeros_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor of zeros with the same shape as the input tensor.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: A tensor of zeros with the same shape.

