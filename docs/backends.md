# Documentation for Backends (backends.py)

## class BroadcastError
BroadcastError exception:<br /><br />Args: None


## class NumpyTensor
ndarray(shape, dtype=float, buffer=None, offset=0,<br />        strides=None, order=None)<br /><br />An array object represents a multidimensional, homogeneous array<br />of fixed-size items.  An associated data-type object describes the<br />format of each element in the array (its byte-order, how many bytes it<br />occupies in memory, whether it is an integer, a floating point number,<br />or something else, etc.)<br /><br />Arrays should be constructed using `array`, `zeros` or `empty` (refer<br />to the See Also section below).  The parameters given here refer to<br />a low-level method (`ndarray(...)`) for instantiating an array.<br /><br />For more information, refer to the `numpy` module and examine the<br />methods and attributes of an array.<br /><br />Parameters<br />----------<br />(for the __new__ method; see Notes below)<br /><br />shape : tuple of ints<br />    Shape of created array.<br />dtype : data-type, optional<br />    Any object that can be interpreted as a numpy data type.<br />buffer : object exposing buffer interface, optional<br />    Used to fill the array with data.<br />offset : int, optional<br />    Offset of array data in buffer.<br />strides : tuple of ints, optional<br />    Strides of data in memory.<br />order : {'C', 'F'}, optional<br />    Row-major (C-style) or column-major (Fortran-style) order.<br /><br />Attributes<br />----------<br />T : ndarray<br />    Transpose of the array.<br />data : buffer<br />    The array's elements, in memory.<br />dtype : dtype object<br />    Describes the format of the elements in the array.<br />flags : dict<br />    Dictionary containing information related to memory use, e.g.,<br />    'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.<br />flat : numpy.flatiter object<br />    Flattened version of the array as an iterator.  The iterator<br />    allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for<br />    assignment examples; TODO).<br />imag : ndarray<br />    Imaginary part of the array.<br />real : ndarray<br />    Real part of the array.<br />size : int<br />    Number of elements in the array.<br />itemsize : int<br />    The memory use of each array element in bytes.<br />nbytes : int<br />    The total number of bytes required to store the array data,<br />    i.e., ``itemsize * size``.<br />ndim : int<br />    The array's number of dimensions.<br />shape : tuple of ints<br />    Shape of the array.<br />strides : tuple of ints<br />    The step-size required to move from one element to the next in<br />    memory. For example, a contiguous ``(3, 4)`` array of type<br />    ``int16`` in C-order has strides ``(8, 2)``.  This implies that<br />    to move from element to element in memory requires jumps of 2 bytes.<br />    To move from row-to-row, one needs to jump 8 bytes at a time<br />    (``2 * 4``).<br />ctypes : ctypes object<br />    Class containing properties of the array needed for interaction<br />    with ctypes.<br />base : ndarray<br />    If the array is a view into another array, that array is its `base`<br />    (unless that array is also a view).  The `base` array is where the<br />    array data is actually stored.<br /><br />See Also<br />--------<br />array : Construct an array.<br />zeros : Create an array, each element of which is zero.<br />empty : Create an array, but leave its allocated memory unchanged (i.e.,<br />        it contains "garbage").<br />dtype : Create a data-type.<br /><br />Notes<br />-----<br />There are two modes of creating an array using ``__new__``:<br /><br />1. If `buffer` is None, then only `shape`, `dtype`, and `order`<br />   are used.<br />2. If `buffer` is an object exposing the buffer interface, then<br />   all keywords are interpreted.<br /><br />No ``__init__`` method is needed because the array is fully initialized<br />after the ``__new__`` method.<br /><br />Examples<br />--------<br />These examples illustrate the low-level `ndarray` constructor.  Refer<br />to the `See Also` section above for easier ways of constructing an<br />ndarray.<br /><br />First mode, `buffer` is None:<br /><br />>>> np.ndarray(shape=(2,2), dtype=float, order='F')<br />array([[ -1.13698227e+002,   4.25087011e-303],<br />       [  2.88528414e-306,   3.27025015e-309]])         #random<br /><br />Second mode:<br /><br />>>> np.ndarray((2,), buffer=np.array([1,2,3]),<br />...            offset=np.int_().itemsize,<br />...            dtype=int) # offset = 1*itemsize, i.e. skip first element<br />array([2, 3])


## class Iterable
Abstract base class for generic types.<br /><br />A generic type is typically declared by inheriting from<br />this class parameterized with one or more type variables.<br />For example, a generic mapping type might be defined as::<br /><br />  class Mapping(Generic[KT, VT]):<br />      def __getitem__(self, key: KT) -> VT:<br />          ...<br />      # Etc.<br /><br />This class can then be used as follows::<br /><br />  def lookup_name(mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:<br />      try:<br />          return mapping[key]<br />      except KeyError:<br />          return default


## class float32
32-bit floating-point number. Character code 'f'. C float compatible.


## class Tuple
Tuple type; Tuple[X, Y] is the cross-product type of X and Y.<br /><br />Example: Tuple[T1, T2] is a tuple of two elements corresponding<br />to type variables T1 and T2.  Tuple[int, float, str] is a tuple<br />of an int, a float and a string.<br /><br />To specify a variable-length tuple of homogeneous type, use Tuple[T, ...].


## class Dict
dict() -> new empty dictionary<br />dict(mapping) -> new dictionary initialized from a mapping object's<br />    (key, value) pairs<br />dict(iterable) -> new dictionary initialized as if via:<br />    d = {}<br />    for k, v in iterable:<br />        d[k] = v<br />dict(**kwargs) -> new dictionary initialized with the name=value pairs<br />    in the keyword argument list.  For example:  dict(one=1, two=2)


## functions

### accumulate
```py

def accumulate(func, a)

```



Accumulates the result of a function over iterable a.<br /><br />For example:<br /><br />'''<br />from collections import namedtuple<br /><br />def square(x):<br /> ~ return x**2<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(1,2)<br />b = accumulate(square, a) # 5<br /><br />a = list(a)<br />b = accumulate(add, a) # 5<br /><br />'''<br /><br />Args:<br /> ~ func (callable): a function with one argument<br /> ~ a (iterable: e.g., list or named tuple)<br /><br />Returns:<br /> ~ float


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


### add\_
```py

def add_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Add tensor a to tensor b using broadcasting.<br /><br />Notes:<br /> ~ Modifies b in place.<br /><br />Args:<br /> ~ a: A tensor<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ None


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


### apply
```py

def apply(func, a)

```



Applies a function over iterable a, giving back an<br />object of the same type as a. That is, b[i] = func(a[i]).<br /><br />For example:<br /><br />'''<br />from collections import namedtuple<br />from operator import mul<br />from cytoolz import partial<br /><br /># create a function to divide by 2<br />halve = partial(mul, 0.5)<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(1,2)<br />b = apply(halve, a) # coordinates(x=0.5, y=1.0)<br /><br />a = list(a)<br />b = apply(halve, a) # [0.5,1.0]<br /><br />'''<br /><br />Args:<br /> ~ func (callable): a function with a single argument<br /> ~ a (iterable: e.g., list or named tuple)<br /><br />Returns:<br /> ~ object of type(a)


### apply\_
```py

def apply_(func_, a)

```



Applies an in place function over iterable a.<br /><br />That is, a[i] = func(a[i]).<br /><br />For example:<br /><br />'''<br />from collections import namedtuple<br />import numpy as np<br />import numexpr as ne<br /><br /># create an in place function to divide an array by 2<br />def halve_(x: np.ndarray) -> None:<br /> ~ ne.evaluate('0.5 * x', out=x)<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(np.ones(1), 2 * np.ones(1))<br />apply_(halve_, a) # a = coordinates(x=np.array(0.5), y=np.array(1.0))<br /><br />a = list(a)<br />apply_(halve_, a) # a = [np.array(0.25), np.array(0.5)]<br /><br />'''<br /><br />Args:<br /> ~ func_ (callable): an in place function of a single argument<br /> ~ a (iterable: e.g., list or named tuple)<br /><br />Returns:<br /> ~ None


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


### argsort
```py

def argsort(x: numpy.ndarray, axis: int=None) -> numpy.ndarray

```



Get the indices of a sorted tensor.<br /><br />Args:<br /> ~ x: A tensor:<br /> ~ axis: The axis of interest.<br /><br />Returns:<br /> ~ tensor (of ints): indices of sorted tensor


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


### center
```py

def center(x: numpy.ndarray, axis: int=0) -> numpy.ndarray

```



Remove the mean along axis.<br /><br />Args:<br /> ~ tensor (num_samples, num_units): the array to center<br /> ~ axis (int; optional): the axis to center along<br /><br />Returns:<br /> ~ tensor (num_samples, num_units)


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


### copy\_tensor
```py

def copy_tensor(tensor: numpy.ndarray) -> numpy.ndarray

```



Copy a tensor.<br /><br />Args:<br /> ~ tensor<br /><br />Returns:<br /> ~ copy of tensor


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


### cov
```py

def cov(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the cross covariance between tensors x and y.<br /><br />Args:<br /> ~ x (tensor (num_samples, num_units_x))<br /> ~ y (tensor (num_samples, num_units_y))<br /><br />Returns:<br /> ~ tensor (num_units_x, num_units_y)


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



Divide tensor b by tensor a using broadcasting.<br /><br />Args:<br /> ~ a: A tensor (non-zero)<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ tensor: b / a


### divide\_
```py

def divide_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Divide tensor b by tensor a using broadcasting.<br /><br />Notes:<br /> ~ Modifies b in place.<br /><br />Args:<br /> ~ a: A tensor (non-zero)<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ tensor: b / a


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


### energy\_distance
```py

def energy_distance(x: numpy.ndarray, y: numpy.ndarray) -> float

```



Compute an energy distance between two tensors treating the rows as observations.<br /><br />Args:<br /> ~ x (tensor (num_samples_1, num_units))<br /> ~ y (tensor (num_samples_2, num_units))<br /><br />Returns:<br /> ~ float: energy distance.<br /><br />Szekely, G.J. (2002)<br />E-statistics: The Energy of Statistical Samples.<br />Technical Report BGSU No 02-16.


### equal
```py

def equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise if two tensors are equal.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor (of bools): Elementwise test of equality between x and y.


### erf
```py

def erf(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise error function of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise error function


### erfinv
```py

def erfinv(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise error function of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise error function


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


### fill\_diagonal\_
```py

def fill_diagonal_(mat: numpy.ndarray, val: Union[int, float]) -> numpy.ndarray

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


### index\_select
```py

def index_select(mat: numpy.ndarray, index: numpy.ndarray, dim: int=0) -> numpy.ndarray

```



Select the specified indices of a tensor along dimension dim.<br />For example, dim = 1 is equivalent to mat[:, index] in numpy.<br /><br />Args:<br /> ~ mat (tensor (num_samples, num_units))<br /> ~ index (tensor; 1 -dimensional)<br /> ~ dim (int)<br /><br />Returns:<br /> ~ if dim == 0:<br /> ~  ~ mat[index, :]<br /> ~ if dim == 1:<br /> ~  ~ mat[:, index]


### int\_tensor
```py

def int_tensor(tensor: numpy.ndarray) -> numpy.ndarray

```



Cast tensor to an int tensor.<br /><br />Args:<br /> ~ tensor: A tensor.<br /><br />Returns:<br /> ~ tensor: Tensor converted to int.


### inv
```py

def inv(mat: numpy.ndarray) -> numpy.ndarray

```



Compute matrix inverse.<br /><br />Args:<br /> ~ mat: A square matrix.<br /><br />Returns:<br /> ~ tensor: The matrix inverse.


### is\_tensor
```py

def is_tensor(x: Union[numpy.float32, numpy.ndarray]) -> bool

```



Test if x is a tensor.<br /><br />Args:<br /> ~ x (float or tensor)<br /><br />Returns:<br /> ~ bool


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


### logdet
```py

def logdet(mat: numpy.ndarray) -> float

```



Compute the logarithm of the determinant of a square matrix.<br /><br />Args:<br /> ~ mat: A square matrix.<br /><br />Returns:<br /> ~ logdet: The logarithm of the matrix determinant.


### logit
```py

def logit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logit function of a tensor. Inverse of the expit function.<br /><br />Args:<br /> ~ x (between 0 and 1): A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise logit function


### mapzip
```py

def mapzip(func, a, b)

```



Applies a function over the zip of iterables a and b,<br />giving back an object of the same type as a. That is,<br />c[i] = func(a[i], b[i]).<br /><br />For example:<br /><br />```<br />from collections import namedtuple<br />from operator import add<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(1,2)<br />b = coords(2,3)<br /><br />c = mapzip(add, a, b) # coordinates(x=2, y=4)<br /><br />a = list(a)<br />b = list(b)<br /><br />c = mapzip(add, a, b) # [2, 4]<br />```<br /><br />Args:<br /> ~ func (callable): a function with two arguments<br /> ~ a (iterable; e.g., list or namedtuple)<br /> ~ b (iterable; e.g., list or namedtuple)<br /><br />Returns:<br /> ~ object of type(a)


### mapzip\_
```py

def mapzip_(func_, a, b)

```



Applies an in place function over the zip of iterables a and b,<br />func(a[i], b[i]).<br /><br />For example:<br /><br />```<br />from collections import namedtuple<br />import numpy as np<br />import numexpr as ne<br /><br />def add_(x: np.ndarray, y: np.ndarray) -> None:<br /> ~ ne.evaluate('x + y', out=x)<br /><br />coords = namedtuple("coordinates", ["x", "y"])<br /><br />a = coords(np.array([1]), np.array([2]))<br />b = coords(np.array([3]), np.array([4]))<br /><br />mapzip_(add_, a, b) # a = coordinates(x=4, y=6)<br /><br />a = list(a)<br />b = list(b)<br /><br />mapzip_(add_, a, b) # a = [7, 10]<br />```<br /><br />Args:<br /> ~ func (callable): an in place function with two arguments<br /> ~ a (iterable; e.g., list or namedtuple)<br /> ~ b (iterable; e.g., list or namedtuple)<br /><br />Returns:<br /> ~ None


### maximum
```py

def maximum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise maximum of two tensors.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise maximum of x and y.


### mean
```py

def mean(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the mean of the elements of a tensor along the specified axis.<br /><br />Args:<br /> ~ x: A float or tensor of rank=2.<br /> ~ axis (optional): The axis for taking the mean.<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is None:<br /> ~  ~ float: The overall mean of the elements in the tensor<br /> ~ else:<br /> ~  ~ tensor: The mean of the tensor along the specified axis.


### minimum
```py

def minimum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise minimum of two tensors.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ y: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise minimum of x and y.


### mix
```py

def mix(w: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute a weighted average of two matrices (x and y) and return the result.<br />Multilinear interpolation.<br /><br />Note:<br /> ~ Modifies x in place.<br /><br />Args:<br /> ~ w: The mixing coefficient tensor between 0 and 1 .<br /> ~ x: A tensor.<br /> ~ y: A tensor:<br /><br />Returns:<br /> ~ tensor = w * x + (1-w) * y


### mix\_inplace
```py

def mix_inplace(w: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> None

```



Compute a weighted average of two matrices (x and y) and store the results in x.<br />Useful for keeping track of running averages during training.<br /><br />x <- w * x + (1-w) * y<br /><br />Note:<br /> ~ Modifies x in place.<br /><br />Args:<br /> ~ w: The mixing coefficient tensor between 0 and 1 .<br /> ~ x: A tensor.<br /> ~ y: A tensor:<br /><br />Returns:<br /> ~ None


### multiply
```py

def multiply(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Multiply tensor b with tensor a using broadcasting.<br /><br />Args:<br /> ~ a: A tensor<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ tensor: a * b


### multiply\_
```py

def multiply_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Multiply tensor b with tensor a using broadcasting.<br /><br />Notes:<br /> ~ Modifies b in place.<br /><br />Args:<br /> ~ a: A tensor<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ None


### ndim
```py

def ndim(tensor: numpy.ndarray) -> int

```



Return the number of dimensions of a tensor.<br /><br />Args:<br /> ~ tensor: A tensor:<br /><br />Returns:<br /> ~ int: The number of dimensions of the tensor.


### norm
```py

def norm(x: numpy.ndarray, axis: int=None, keepdims: bool=False) -> Union[numpy.float32, numpy.ndarray]

```



Return the L2 norm of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ axis (optional): the axis for taking the norm<br /> ~ keepdims (optional): If this is set to true, the dimension of the tensor<br /> ~  ~  ~  ~  ~  ~  is unchanged. Otherwise, the reduced axis is removed<br /> ~  ~  ~  ~  ~  ~  and the dimension of the array is 1 less.<br /><br />Returns:<br /> ~ if axis is none:<br /> ~  ~ float: The L2 norm of the tensor<br /> ~  ~    (i.e., the sqrt of the sum of the squared elements).<br /> ~ else:<br /> ~  ~ tensor: The L2 norm along the specified axis.


### normal\_cdf
```py

def normal_cdf(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise cumulative distribution function of the standard normal distribution.<br /><br />For the CDF of a normal distributon with mean u and standard deviation sigma, use<br />normal_cdf((x-u)/sigma).<br /><br />Args:<br /> ~ x (tensor)<br /><br />Returns:<br /> ~ tensor: Elementwise cdf


### normal\_inverse\_cdf
```py

def normal_inverse_cdf(p: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse cumulative distribution function of the standard normal<br />distribution.<br /><br />For the inverse CDF of a normal distributon with mean u and standard deviation sigma,<br />use u + sigma * normal_inverse_cdf(p).<br /><br />Args:<br /> ~ p (tensor bounded in (0,1))<br /><br />Returns:<br /> ~ tensor: Elementwise inverse cdf


### normal\_pdf
```py

def normal_pdf(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise probability density function of the standard normal distribution.<br /><br />For the PDF of a normal distributon with mean u and standard deviation sigma, use<br />normal_pdf((x-u)/sigma) / sigma.<br /><br />Args:<br /> ~ x (tensor)<br /><br />Returns:<br /> ~ tensor: Elementwise pdf


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


### num\_elements
```py

def num_elements(tensor: numpy.ndarray) -> int

```



Return the number of elements in a tensor.<br /><br />Args:<br /> ~ tensor: A tensor:<br /><br />Returns:<br /> ~ int: The number of elements in the tensor.


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


### pdist
```py

def pdist(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the pairwise distance matrix between the rows of x and y.<br /><br />Args:<br /> ~ x (tensor (num_samples_1, num_units))<br /> ~ y (tensor (num_samples_2, num_units))<br /><br />Returns:<br /> ~ tensor (num_samples_1, num_samples_2)


### pinv
```py

def pinv(mat: numpy.ndarray) -> numpy.ndarray

```



Compute matrix pseudoinverse.<br /><br />Args:<br /> ~ mat: A square matrix.<br /><br />Returns:<br /> ~ tensor: The matrix pseudoinverse.


### qr
```py

def qr(mat: numpy.ndarray) -> Tuple[numpy.ndarray]

```



Compute the QR decomposition of a matrix.<br />The QR decomposition factorizes a matrix A into a product<br />A = QR of an orthonormal matrix Q and an upper triangular matrix R.<br />Provides an orthonormalization of the columns of the matrix.<br /><br />Args:<br /> ~ mat: A matrix.<br /><br />Returns:<br /> ~ (Q, R): Tuple of tensors.


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


### rand\_like
```py

def rand_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Generate a tensor of the same shape as the specified tensor<br /><br />Args:<br /> ~ tensor: tensor with desired shape.<br /><br />Returns:<br /> ~ tensor: Random numbers between 0 and 1.


### rand\_softmax
```py

def rand_softmax(phi: numpy.ndarray) -> numpy.ndarray

```



Draw random 1-hot samples according to softmax probabilities.<br /><br />Given an effective field vector v,<br />the softmax probabilities are p = exp(v) / sum(exp(v))<br /><br />A 1-hot vector x is sampled according to p.<br /><br />Args:<br /> ~ phi (tensor (batch_size, num_units)): the effective field<br /><br />Returns:<br /> ~ tensor (batch_size, num_units): random 1-hot samples<br /> ~  ~ from the softmax distribution.


### randn
```py

def randn(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with random numbers<br />drawn from a standard normal distribution (mean = 0, variance = 1).<br /><br />Args:<br /> ~ shape: Desired shape of the random tensor.<br /><br />Returns:<br /> ~ tensor: Random numbers between from a standard normal distribution.


### randn\_like
```py

def randn_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Generate a tensor of the same shape as the specified tensor<br />filled with normal(0,1) random numbers<br /><br />Args:<br /> ~ tensor: tensor with desired shape.<br /><br />Returns:<br /> ~ tensor: Random numbers between from a standard normal distribution.


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


### scatter\_
```py

def scatter_(mat: numpy.ndarray, inds: numpy.ndarray, val: Union[int, float]) -> numpy.ndarray

```



Assign a value a specific points in a matrix.<br />Iterates along the rows of mat,<br />successively assigning val to column indices given by inds.<br /><br />Note:<br /> ~ Modifies mat in place.<br /><br />Args:<br /> ~ mat: A tensor.<br /> ~ inds: The indices<br /> ~ val: The value to insert


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


### softmax
```py

def softmax(x: numpy.ndarray) -> numpy.ndarray

```



Softmax function on a tensor.<br />Exponentiaties the tensor elementwise and divides<br /> ~ by the sum along axis=1.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Softmax of the tensor.


### softplus
```py

def softplus(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise softplus function of a tensor.<br /><br />Args:<br /> ~ x: A tensor.<br /><br />Returns:<br /> ~ tensor: Elementwise softplus.


### sort
```py

def sort(x: numpy.ndarray, axis: int=None) -> numpy.ndarray

```



Sort a tensor along the specied axis.<br /><br />Args:<br /> ~ x: A tensor:<br /> ~ axis: The axis of interest.<br /><br />Returns:<br /> ~ tensor (of floats): sorted tensor


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

def square_mix_inplace(w: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> None

```



Compute a weighted average of two matrices (x and y^2) and store the results in x.<br />Useful for keeping track of running averages of squared matrices during training.<br /><br />x <- w x + (1-w) * y**2<br /><br />Note:<br /> ~ Modifies x in place.<br /><br />Args:<br /> ~ w: The mixing coefficient tensor between 0 and 1.<br /> ~ x: A tensor.<br /> ~ y: A tensor:<br /><br />Returns:<br /> ~ None


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


### subtract\_
```py

def subtract_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Subtract tensor a from tensor b using broadcasting.<br /><br />Notes:<br /> ~ Modifies b in place.<br /><br />Args:<br /> ~ a: A tensor<br /> ~ b: A tensor<br /><br />Returns:<br /> ~ None


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


### tclip
```py

def tclip(tensor: numpy.ndarray, a_min: numpy.ndarray=None, a_max: numpy.ndarray=None) -> numpy.ndarray

```



Return a tensor with its values clipped element-wise between a_min and a_max tensors.<br />The implementation is identical to clip.<br /><br />Args:<br /> ~ tensor: A tensor.<br /> ~ a_min (optional tensor): The desired lower bound on the elements of the tensor.<br /> ~ a_max (optional tensor): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br /> ~ tensor: A new tensor with its values clipped between a_min and a_max.


### tclip\_inplace
```py

def tclip_inplace(tensor: numpy.ndarray, a_min: numpy.ndarray=None, a_max: numpy.ndarray=None) -> None

```



Clip the values of a tensor elementwise between a_min and a_max tensors.<br />The implementation is identical to tclip_inplace<br /><br />Note:<br /> ~ Modifies tensor in place.<br /><br />Args:<br /> ~ tensor: A tensor.<br /> ~ a_min (optional tensor): The desired lower bound on the elements of the tensor.<br /> ~ a_max (optional tessor): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br /> ~ None


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


### tmul
```py

def tmul(a: Union[int, float], x: numpy.ndarray) -> numpy.ndarray

```



Elementwise multiplication of tensor x by scalar a.<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ a: scalar.<br /><br />Returns:<br /> ~ tensor: Elementwise a * x.


### tmul\_
```py

def tmul_(a: Union[int, float], x: numpy.ndarray) -> numpy.ndarray

```



Elementwise multiplication of tensor x by scalar a.<br /><br />Notes:<br /> ~ Modifes x in place<br /><br />Args:<br /> ~ x: A tensor.<br /> ~ a: scalar.<br /><br />Returns:<br /> ~ tensor: Elementwise a * x.


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


### unsqueeze
```py

def unsqueeze(tensor: numpy.ndarray, axis: int) -> numpy.ndarray

```



Return tensor with a new axis inserted.<br /><br />Args:<br /> ~ tensor: A tensor.<br /> ~ axis: The desired axis.<br /><br />Returns:<br /> ~ tensor: A tensor with the new axis inserted.


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

