# Documentation for Select_Backend (select_backend.py)

## class BroadcastError
BroadcastError exception:<br /><br />Args: None


## class NumpyTensor
ndarray(shape, dtype=float, buffer=None, offset=0,<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;strides=None, order=None)<br /><br />An array object represents a multidimensional, homogeneous array<br />of fixed-size items.  An associated data-type object describes the<br />format of each element in the array (its byte-order, how many bytes it<br />occupies in memory, whether it is an integer, a floating point number,<br />or something else, etc.)<br /><br />Arrays should be constructed using `array`, `zeros` or `empty` (refer<br />to the See Also section below).  The parameters given here refer to<br />a low-level method (`ndarray(...)`) for instantiating an array.<br /><br />For more information, refer to the `numpy` module and examine the<br />methods and attributes of an array.<br /><br />Parameters<br />----------<br />(for the __new__ method; see Notes below)<br /><br />shape : tuple of ints<br />&nbsp;&nbsp;&nbsp;&nbsp;Shape of created array.<br />dtype : data-type, optional<br />&nbsp;&nbsp;&nbsp;&nbsp;Any object that can be interpreted as a numpy data type.<br />buffer : object exposing buffer interface, optional<br />&nbsp;&nbsp;&nbsp;&nbsp;Used to fill the array with data.<br />offset : int, optional<br />&nbsp;&nbsp;&nbsp;&nbsp;Offset of array data in buffer.<br />strides : tuple of ints, optional<br />&nbsp;&nbsp;&nbsp;&nbsp;Strides of data in memory.<br />order : {'C', 'F'}, optional<br />&nbsp;&nbsp;&nbsp;&nbsp;Row-major (C-style) or column-major (Fortran-style) order.<br /><br />Attributes<br />----------<br />T : ndarray<br />&nbsp;&nbsp;&nbsp;&nbsp;Transpose of the array.<br />data : buffer<br />&nbsp;&nbsp;&nbsp;&nbsp;The array's elements, in memory.<br />dtype : dtype object<br />&nbsp;&nbsp;&nbsp;&nbsp;Describes the format of the elements in the array.<br />flags : dict<br />&nbsp;&nbsp;&nbsp;&nbsp;Dictionary containing information related to memory use, e.g.,<br />&nbsp;&nbsp;&nbsp;&nbsp;'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.<br />flat : numpy.flatiter object<br />&nbsp;&nbsp;&nbsp;&nbsp;Flattened version of the array as an iterator.  The iterator<br />&nbsp;&nbsp;&nbsp;&nbsp;allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for<br />&nbsp;&nbsp;&nbsp;&nbsp;assignment examples; TODO).<br />imag : ndarray<br />&nbsp;&nbsp;&nbsp;&nbsp;Imaginary part of the array.<br />real : ndarray<br />&nbsp;&nbsp;&nbsp;&nbsp;Real part of the array.<br />size : int<br />&nbsp;&nbsp;&nbsp;&nbsp;Number of elements in the array.<br />itemsize : int<br />&nbsp;&nbsp;&nbsp;&nbsp;The memory use of each array element in bytes.<br />nbytes : int<br />&nbsp;&nbsp;&nbsp;&nbsp;The total number of bytes required to store the array data,<br />&nbsp;&nbsp;&nbsp;&nbsp;i.e., ``itemsize * size``.<br />ndim : int<br />&nbsp;&nbsp;&nbsp;&nbsp;The array's number of dimensions.<br />shape : tuple of ints<br />&nbsp;&nbsp;&nbsp;&nbsp;Shape of the array.<br />strides : tuple of ints<br />&nbsp;&nbsp;&nbsp;&nbsp;The step-size required to move from one element to the next in<br />&nbsp;&nbsp;&nbsp;&nbsp;memory. For example, a contiguous ``(3, 4)`` array of type<br />&nbsp;&nbsp;&nbsp;&nbsp;``int16`` in C-order has strides ``(8, 2)``.  This implies that<br />&nbsp;&nbsp;&nbsp;&nbsp;to move from element to element in memory requires jumps of 2 bytes.<br />&nbsp;&nbsp;&nbsp;&nbsp;To move from row-to-row, one needs to jump 8 bytes at a time<br />&nbsp;&nbsp;&nbsp;&nbsp;(``2 * 4``).<br />ctypes : ctypes object<br />&nbsp;&nbsp;&nbsp;&nbsp;Class containing properties of the array needed for interaction<br />&nbsp;&nbsp;&nbsp;&nbsp;with ctypes.<br />base : ndarray<br />&nbsp;&nbsp;&nbsp;&nbsp;If the array is a view into another array, that array is its `base`<br />&nbsp;&nbsp;&nbsp;&nbsp;(unless that array is also a view).  The `base` array is where the<br />&nbsp;&nbsp;&nbsp;&nbsp;array data is actually stored.<br /><br />See Also<br />--------<br />array : Construct an array.<br />zeros : Create an array, each element of which is zero.<br />empty : Create an array, but leave its allocated memory unchanged (i.e.,<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;it contains "garbage").<br />dtype : Create a data-type.<br /><br />Notes<br />-----<br />There are two modes of creating an array using ``__new__``:<br /><br />1. If `buffer` is None, then only `shape`, `dtype`, and `order`<br />   are used.<br />2. If `buffer` is an object exposing the buffer interface, then<br />   all keywords are interpreted.<br /><br />No ``__init__`` method is needed because the array is fully initialized<br />after the ``__new__`` method.<br /><br />Examples<br />--------<br />These examples illustrate the low-level `ndarray` constructor.  Refer<br />to the `See Also` section above for easier ways of constructing an<br />ndarray.<br /><br />First mode, `buffer` is None:<br /><br />>>> np.ndarray(shape=(2,2), dtype=float, order='F')<br />array([[ -1.13698227e+002,   4.25087011e-303],<br />&nbsp;&nbsp;&nbsp;&nbsp;   [  2.88528414e-306,   3.27025015e-309]])&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; #random<br /><br />Second mode:<br /><br />>>> np.ndarray((2,), buffer=np.array([1,2,3]),<br />...&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;offset=np.int_().itemsize,<br />...&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dtype=int) # offset = 1*itemsize, i.e. skip first element<br />array([2, 3])


## class Double
Double-precision floating-point number type, compatible with Python `float`<br />and C ``double``.<br />Character code: ``'d'``.<br />Canonical name: ``np.double``.<br />Alias: ``np.float_``.<br />Alias *on this platform*: ``np.float64``: 64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa.


## class Dtype
dtype(obj, align=False, copy=False)<br /><br />Create a data type object.<br /><br />A numpy array is homogeneous, and contains elements described by a<br />dtype object. A dtype object can be constructed from different<br />combinations of fundamental numeric types.<br /><br />Parameters<br />----------<br />obj<br />&nbsp;&nbsp;&nbsp;&nbsp;Object to be converted to a data type object.<br />align : bool, optional<br />&nbsp;&nbsp;&nbsp;&nbsp;Add padding to the fields to match what a C compiler would output<br />&nbsp;&nbsp;&nbsp;&nbsp;for a similar C-struct. Can be ``True`` only if `obj` is a dictionary<br />&nbsp;&nbsp;&nbsp;&nbsp;or a comma-separated string. If a struct dtype is being created,<br />&nbsp;&nbsp;&nbsp;&nbsp;this also sets a sticky alignment flag ``isalignedstruct``.<br />copy : bool, optional<br />&nbsp;&nbsp;&nbsp;&nbsp;Make a new copy of the data-type object. If ``False``, the result<br />&nbsp;&nbsp;&nbsp;&nbsp;may just be a reference to a built-in data-type object.<br /><br />See also<br />--------<br />result_type<br /><br />Examples<br />--------<br />Using array-scalar type:<br /><br />>>> np.dtype(np.int16)<br />dtype('int16')<br /><br />Structured type, one field name 'f1', containing int16:<br /><br />>>> np.dtype([('f1', np.int16)])<br />dtype([('f1', '<i2')])<br /><br />Structured type, one field named 'f1', in itself containing a structured<br />type with one field:<br /><br />>>> np.dtype([('f1', [('f1', np.int16)])])<br />dtype([('f1', [('f1', '<i2')])])<br /><br />Structured type, two fields: the first field contains an unsigned int, the<br />second an int32:<br /><br />>>> np.dtype([('f1', np.uint), ('f2', np.int32)])<br />dtype([('f1', '<u4'), ('f2', '<i4')])<br /><br />Using array-protocol type strings:<br /><br />>>> np.dtype([('a','f8'),('b','S10')])<br />dtype([('a', '<f8'), ('b', '|S10')])<br /><br />Using comma-separated field formats.  The shape is (2,3):<br /><br />>>> np.dtype("i4, (2,3)f8")<br />dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])<br /><br />Using tuples.  ``int`` is a fixed type, 3 the field's shape.  ``void``<br />is a flexible type, here of size 10:<br /><br />>>> np.dtype([('hello',(int,3)),('world',np.void,10)])<br />dtype([('hello', '<i4', 3), ('world', '|V10')])<br /><br />Subdivide ``int16`` into 2 ``int8``'s, called x and y.  0 and 1 are<br />the offsets in bytes:<br /><br />>>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))<br />dtype(('<i2', [('x', '|i1'), ('y', '|i1')]))<br /><br />Using dictionaries.  Two fields named 'gender' and 'age':<br /><br />>>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})<br />dtype([('gender', '|S1'), ('age', '|u1')])<br /><br />Offsets in bytes, here 0 and 25:<br /><br />>>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})<br />dtype([('surname', '|S25'), ('age', '|u1')])


## class Float
Single-precision floating-point number type, compatible with C ``float``.<br />Character code: ``'f'``.<br />Canonical name: ``np.single``.<br />Alias *on this platform*: ``np.float32``: 32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa.


## class Byte
Unsigned integer type, compatible with C ``unsigned char``.<br />Character code: ``'B'``.<br />Canonical name: ``np.ubyte``.<br />Alias *on this platform*: ``np.uint8``: 8-bit unsigned integer (0 to 255).


## class Long
Signed integer type, compatible with Python `int` anc C ``long``.<br />Character code: ``'l'``.<br />Canonical name: ``np.int_``.<br />Alias *on this platform*: ``np.int64``: 64-bit signed integer (-9223372036854775808 to 9223372036854775807).<br />Alias *on this platform*: ``np.intp``: Signed integer large enough to fit pointer, compatible with C ``intptr_t``.


## class Int
Signed integer type, compatible with C ``int``.<br />Character code: ``'i'``.<br />Canonical name: ``np.intc``.<br />Alias *on this platform*: ``np.int32``: 32-bit signed integer (-2147483648 to 2147483647).


## functions

### acosh
```py

def acosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (greater than 1): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise inverse hyperbolic cosine.


### add
```py

def add(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Add tensor a to tensor b using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: a + b


### add\_
```py

def add_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Add tensor a to tensor b using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### affine
```py

def affine(a: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray

```



Evaluate the affine transformation a + W b.<br /><br />a ~ vector, b ~ vector, W ~ matrix:<br />a_i + \sum_j W_ij b_j<br /><br />a ~ matrix, b ~ matrix, W ~ matrix:<br />a_ij + \sum_k W_ik b_kj<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor (1 or 2 dimensional).<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor (1 or 2 dimensional).<br />&nbsp;&nbsp;&nbsp;&nbsp;W: A tensor (2 dimensional).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Affine transformation a + W b.


### allclose
```py

def allclose(x: numpy.ndarray, y: numpy.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool

```



Test if all elements in the two tensors are approximately equal.<br /><br />absolute(x - y) <= (atol + rtol * absolute(y))<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;rtol (optional): Relative tolerance.<br />&nbsp;&nbsp;&nbsp;&nbsp;atol (optional): Absolute tolerance.<br /><br />returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: Check if all of the elements in the tensors are approximately equal.


### argmax
```py

def argmax(x: numpy.ndarray, axis: int) -> numpy.ndarray

```



Compute the indices of the maximal elements in x along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of ints): Indices of the maximal elements in x along the<br />&nbsp;&nbsp;&nbsp;&nbsp;specified axis.


### argmin
```py

def argmin(x: numpy.ndarray, axis: int) -> numpy.ndarray

```



Compute the indices of the minimal elements in x along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of ints): Indices of the minimum elements in x along the<br />&nbsp;&nbsp;&nbsp;&nbsp;specified axis.


### argsort
```py

def argsort(x: numpy.ndarray, axis: int = None) -> numpy.ndarray

```



Get the indices of a sorted tensor.<br />If axis=None this flattens x<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of ints): indices of sorted tensor


### atanh
```py

def atanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic tangent of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (between -1 and +1): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise inverse hyperbolic tangent


### batch\_dot
```py

def batch_dot(a: numpy.ndarray, b: numpy.ndarray, axis: int = 1) -> numpy.ndarray

```



Compute the dot product of vectors batch-wise.<br />Let a be an L x N matrix where each row a_i is a vector.<br />Let b be an L x N matrix where each row b_i is a vector.<br />Then batch_dot(a, b) = \sum_j a_ij * b_ij<br />One can choose the axis to sum along with the axis argument.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (int): The axis to dot along.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.


### batch\_outer
```py

def batch_outer(vis: numpy.ndarray, hid: numpy.ndarray) -> numpy.ndarray

```



Let v by a L x N matrix where each row v_i is a visible vector.<br />Let h be a L x M matrix where each row h_i is a hidden vector.<br />Then, batch_outer(v, h) = \sum_i v_i h_i^T<br />Returns an N x M matrix.<br /><br />The actual computation is performed with a vectorized expression.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vis: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;hid: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A matrix.


### batch\_quadratic
```py

def batch_quadratic(vis: numpy.ndarray, W: numpy.ndarray, hid: numpy.ndarray, axis: int = 1) -> numpy.ndarray

```



Let v by an L x N matrix where each row v_i is a visible vector.<br />Let h be an L x M matrix where each row h_i is a hidden vector.<br />And, let W be a N x M matrix of weights.<br />Then, batch_quadratic(v,W,h) = \sum_i v_i^T W h_i<br /><br />The actual computation is performed with a vectorized expression.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vis: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;W: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;hid: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): Axis of interest<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector.


### broadcast
```py

def broadcast(vec: numpy.ndarray, matrix: numpy.ndarray) -> numpy.ndarray

```



Broadcasts vec into the shape of matrix following numpy rules:<br /><br />vec ~ (N, 1) broadcasts to matrix ~ (N, M)<br />vec ~ (1, N) and (N,) broadcast to matrix ~ (M, N)<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vec: A vector (either flat, row, or column).<br />&nbsp;&nbsp;&nbsp;&nbsp;matrix: A matrix (i.e., a 2D tensor).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of the same size as matrix containing the elements<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of the vector.<br /><br />Raises:<br />&nbsp;&nbsp;&nbsp;&nbsp;BroadcastError


### cast\_float
```py

def cast_float(tensor: numpy.ndarray) -> numpy.ndarray

```



Cast tensor to a float tensor.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;If tensor is already float, no copy is made.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: copy of tensor converted to floating point.


### cast\_long
```py

def cast_long(tensor: numpy.ndarray) -> numpy.ndarray

```



Cast tensor to an long int tensor.<br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;If tensor is already long int, no copy is made.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Copy of tensor converted to long int.


### center
```py

def center(x: numpy.ndarray, axis: int = 0) -> numpy.ndarray

```



Remove the mean along axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units): the array to center<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (int; optional): the axis to center along<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units)


### clip
```py

def clip(tensor: numpy.ndarray, a_min: Union[int, float] = None, a_max: Union[int, float] = None) -> numpy.ndarray

```



Return a tensor with its values clipped between a_min and a_max.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A new tensor with its values clipped between a_min and a_max.


### clip\_
```py

def clip_(tensor: numpy.ndarray, a_min: Union[int, float] = None, a_max: Union[int, float] = None) -> None

```



Clip the values of a tensor between a_min and a_max.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies tensor in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### copy\_tensor
```py

def copy_tensor(tensor: numpy.ndarray) -> numpy.ndarray

```



Copy a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;copy of tensor


### corr
```py

def corr(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the cross correlation between tensors x and y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor (num_samples, num_units_x))<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor (num_samples, num_units_y))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_units_x, num_units_y)


### cos
```py

def cos(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise cosine.


### cosh
```py

def cosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise hyperbolic cosine.


### cov
```py

def cov(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the cross covariance between tensors x and y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor (num_samples, num_units_x))<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor (num_samples, num_units_y))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_units_x, num_units_y)


### cumsum
```py

def cumsum(x: numpy.ndarray, axis: int = 0) -> numpy.ndarray

```



Return the cumulative sum of elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the sum.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the cumulative sum of elements of the tensor along the specified axis.


### diag
```py

def diag(mat: numpy.ndarray) -> numpy.ndarray

```



Return the diagonal elements of a matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector (i.e., 1D tensor) containing the diagonal<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;elements of mat.


### diagonal\_matrix
```py

def diagonal_matrix(vec: numpy.ndarray) -> numpy.ndarray

```



Return a matrix with vec along the diagonal.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vec: A vector (i.e., 1D tensor).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A matrix with the elements of vec along the diagonal,<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and zeros elsewhere.


### divide
```py

def divide(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Divide tensor b by tensor a using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor (non-zero)<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: b / a


### divide\_
```py

def divide_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Divide tensor b by tensor a using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor (non-zero)<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: b / a


### dot
```py

def dot(a: numpy.ndarray, b: numpy.ndarray) -> Union[numpy.float32, numpy.ndarray]

```



Compute the matrix/dot product of tensors a and b.<br /><br />Vector-Vector:<br />&nbsp;&nbsp;&nbsp;&nbsp;\sum_i a_i b_i<br /><br />Matrix-Vector:<br />&nbsp;&nbsp;&nbsp;&nbsp;\sum_j a_ij b_j<br /><br />Matrix-Matrix:<br />&nbsp;&nbsp;&nbsp;&nbsp;\sum_j a_ij b_jk<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if a and b are 1-dimensions:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: the dot product of vectors a and b<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: the matrix product of tensors a and b


### dtype
```py

def dtype(tensor: numpy.ndarray) -> type

```



Return the type of the tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;type: The type of the elements in the tensor.


### equal
```py

def equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise if two tensors are equal.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of equality between x and y.


### exp
```py

def exp(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise exponential function of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (non-negative): Elementwise exponential.


### expit
```py

def expit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise expit (a.k.a. logistic) function of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise expit (a.k.a. logistic).


### fill\_diagonal\_
```py

def fill_diagonal_(mat: numpy.ndarray, val: Union[int, float]) -> numpy.ndarray

```



Fill the diagonal of the matirx with a specified value.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies mat in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;val: The value to put along the diagonal.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### flatten
```py

def flatten(tensor: Union[numpy.float32, numpy.ndarray]) -> Union[numpy.float32, numpy.ndarray]

```



Return a flattened tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor or scalar.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;result: If arg is a tensor, return a flattened 1D tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If arg is a scalar, return the scalar.


### float\_scalar
```py

def float_scalar(scalar: Union[int, float]) -> float

```



Cast scalar to a 32-bit float.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;scalar: A scalar quantity:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;numpy.float32: Scalar converted to floating point.


### float\_tensor
```py

def float_tensor(tensor: Union[numpy.ndarray, Iterable[float]]) -> numpy.ndarray

```



Construct a float tensor.<br />This will always copy the data in tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A float tensor or list of floats.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Constructed float tensor.


### from\_numpy\_array
```py

def from_numpy_array(tensor: numpy.ndarray) -> numpy.ndarray

```



Construct a tensor from a numpy array. A noop.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A numpy ndarray<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted from ndarray.


### greater
```py

def greater(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x > y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x > y.


### greater\_equal
```py

def greater_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x >= y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x >= y.


### hstack
```py

def hstack(tensors: Iterable[numpy.ndarray]) -> numpy.ndarray

```



Concatenate tensors along the first axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: A list of tensors.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensors stacked along axis=1.


### identity
```py

def identity(n: int) -> numpy.ndarray

```



Return the n-dimensional identity matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n: The desired size of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The n x n identity matrix with ones along the diagonal<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and zeros elsewhere.


### index\_select
```py

def index_select(mat: numpy.ndarray, index: numpy.ndarray, dim: int = 0) -> numpy.ndarray

```



Select the specified indices of a tensor along dimension dim.<br />For example, dim = 1 is equivalent to mat[:, index] in numpy.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat (tensor (num_samples, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;index (tensor; 1 -dimensional)<br />&nbsp;&nbsp;&nbsp;&nbsp;dim (int)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if dim == 0:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mat[index, :]<br />&nbsp;&nbsp;&nbsp;&nbsp;if dim == 1:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mat[:, index]


### inv
```py

def inv(mat: numpy.ndarray) -> numpy.ndarray

```



Compute matrix inverse.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The matrix inverse.


### is\_tensor
```py

def is_tensor(x: Union[numpy.float32, numpy.ndarray]) -> bool

```



Test if x is a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (float or tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool


### lesser
```py

def lesser(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x < y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x < y.


### lesser\_equal
```py

def lesser_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if x <= y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x <= y.


### log
```py

def log(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise natural logarithm of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (non-negative): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise natural logarithm.


### logaddexp
```py

def logaddexp(x1: numpy.ndarray, x2: numpy.ndarray) -> numpy.ndarray

```



Elementwise logaddexp function: log(exp(x1) + exp(x2))<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x1: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;x2: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise logaddexp.


### logcosh
```py

def logcosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logarithm of the hyperbolic cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise logarithm of the hyperbolic cosine.


### logdet
```py

def logdet(mat: numpy.ndarray) -> float

```



Compute the logarithm of the determinant of a square matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;logdet: The logarithm of the matrix determinant.


### logical\_and
```py

def logical_and(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the elementwise logical and on two tensors<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### logical\_not
```py

def logical_not(x: numpy.ndarray) -> numpy.ndarray

```



Invert a logical array (True -> False, False -> True).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### logical\_or
```py

def logical_or(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the elementwise logical or on two tensors<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### logit
```py

def logit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logit function of a tensor. Inverse of the expit function.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (between 0 and 1): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise logit function


### long\_tensor
```py

def long_tensor(tensor: Union[numpy.ndarray, Iterable[int]]) -> numpy.ndarray

```



Construct a long int tensor.<br />This will always copy the data in tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A long tensor or list of longs.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to long int.


### matrix\_sqrt
```py

def matrix_sqrt(mat: numpy.ndarray) -> numpy.ndarray

```



Compute the matrix square root using an SVD<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;logdet: The logarithm of the matrix determinant.


### maximum
```py

def maximum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise maximum of two tensors.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise maximum of x and y.


### mean
```py

def mean(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the mean of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor of rank=2.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the mean.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall mean of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The mean of the tensor along the specified axis.


### minimum
```py

def minimum(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise minimum of two tensors.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise minimum of x and y.


### mix
```py

def mix(w: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute a weighted average of two matrices (x and y) and return the result.<br />Multilinear interpolation.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies x in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;w: The mixing coefficient tensor between 0 and 1 .<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor = w * x + (1-w) * y


### mix\_
```py

def mix_(w: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> None

```



Compute a weighted average of two matrices (x and y) and store the results in x.<br />Useful for keeping track of running averages during training.<br /><br />x <- w * x + (1-w) * y<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies x in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;w: The mixing coefficient tensor between 0 and 1 .<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### multiply
```py

def multiply(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Multiply tensor b with tensor a using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: a * b


### multiply\_
```py

def multiply_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Multiply tensor b with tensor a using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### ndim
```py

def ndim(tensor: numpy.ndarray) -> int

```



Return the number of dimensions of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;int: The number of dimensions of the tensor.


### norm
```py

def norm(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the L2 norm of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): the axis for taking the norm<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is none:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The L2 norm of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   (i.e., the sqrt of the sum of the squared elements).<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The L2 norm along the specified axis.


### normal\_pdf
```py

def normal_pdf(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise probability density function of the standard normal distribution.<br /><br />For the PDF of a normal distributon with mean u and standard deviation sigma, use<br />normal_pdf((x-u)/sigma) / sigma.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise pdf


### normalize
```py

def normalize(x: numpy.ndarray) -> numpy.ndarray

```



Divide x by it's sum.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A non-negative tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor normalized by it's sum.


### not\_equal
```py

def not_equal(x: numpy.ndarray, y: numpy.ndarray) -> Union[bool, numpy.ndarray]

```



Elementwise test if two tensors are not equal.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of non-equality between x and y.


### num\_elements
```py

def num_elements(tensor: numpy.ndarray) -> int

```



Return the number of elements in a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;int: The number of elements in the tensor.


### ones
```py

def ones(shape: Tuple[int], dtype: numpy.dtype = <class 'numpy.float32'>) -> numpy.ndarray

```



Return a tensor of a specified shape filled with ones.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: The shape of the desired tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of ones with the desired shape.


### ones\_like
```py

def ones_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor of ones with the same shape as the input tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor with the same shape.


### outer
```py

def outer(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Compute the outer product of vectors x and y.<br /><br />mat_{ij} = x_i * y_j<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A vector (i.e., a 1D tensor).<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A vector (i.e., a 1D tensor).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Outer product of vectors x and y.


### pinv
```py

def pinv(mat: numpy.ndarray) -> numpy.ndarray

```



Compute matrix pseudoinverse.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The matrix pseudoinverse.


### qr
```py

def qr(mat: numpy.ndarray) -> Tuple[numpy.ndarray]

```



Compute the QR decomposition of a matrix.<br />The QR decomposition factorizes a matrix A into a product<br />A = QR of an orthonormal matrix Q and an upper triangular matrix R.<br />Provides an orthonormalization of the columns of the matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;(Q, R): Tuple of tensors.


### quadratic
```py

def quadratic(a: numpy.ndarray, b: numpy.ndarray, W: numpy.ndarray) -> numpy.ndarray

```



Evaluate the quadratic form a W b.<br /><br />a ~ vector, b ~ vector, W ~ matrix:<br />\sum_ij a_i W_ij b_j<br /><br />a ~ matrix, b ~ matrix, W ~ matrix:<br />\sum_kl a_ik W_kl b_lj<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;W: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Quadratic function a W b.


### rand
```py

def rand(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with uniform random numbers<br />between 0 and 1.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: Desired shape of the random tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between 0 and 1.


### rand\_int
```py

def rand_int(a: int, b: int, shape: Tuple[int]) -> numpy.ndarray

```



Generate random integers in [a, b).<br />Fills a tensor of a given shape<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a (int): the minimum (inclusive) of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;b (int): the maximum (exclusive) of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: the shape of the output tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (shape): the random integer samples.


### rand\_like
```py

def rand_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Generate a tensor of the same shape as the specified tensor<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: tensor with desired shape.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between 0 and 1.


### rand\_samples
```py

def rand_samples(tensor: numpy.ndarray, num: int) -> numpy.ndarray

```



Collect a random number samples from a tensor with replacement.<br />Only supports the input tensor being a vector.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor ((num_samples)): a vector of values.<br />&nbsp;&nbsp;&nbsp;&nbsp;num (int): the number of samples to take.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;samples ((num)): a vector of sampled values.


### rand\_softmax
```py

def rand_softmax(phi: numpy.ndarray) -> numpy.ndarray

```



Draw random 1-hot samples according to softmax probabilities.<br /><br />Given an effective field vector v,<br />the softmax probabilities are p = exp(v) / sum(exp(v))<br /><br />A 1-hot vector x is sampled according to p.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;phi (tensor (batch_size, num_units)): the effective field<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (batch_size, num_units): random 1-hot samples<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;from the softmax distribution.


### rand\_softmax\_units
```py

def rand_softmax_units(phi: numpy.ndarray) -> numpy.ndarray

```



Draw random unit values according to softmax probabilities.<br /><br />Given an effective field vector v,<br />the softmax probabilities are p = exp(v) / sum(exp(v))<br /><br />The unit values (the on-units for a 1-hot encoding)<br />are sampled according to p.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;phi (tensor (batch_size, num_units)): the effective field<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (batch_size,): random unit values from the softmax distribution.


### randn
```py

def randn(shape: Tuple[int]) -> numpy.ndarray

```



Generate a tensor of the specified shape filled with random numbers<br />drawn from a standard normal distribution (mean = 0, variance = 1).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: Desired shape of the random tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between from a standard normal distribution.


### randn\_like
```py

def randn_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Generate a tensor of the same shape as the specified tensor<br />filled with normal(0,1) random numbers<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: tensor with desired shape.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random numbers between from a standard normal distribution.


### reciprocal
```py

def reciprocal(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (non-zero): A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise inverse.


### repeat
```py

def repeat(tensor: numpy.ndarray, n: int) -> numpy.ndarray

```



Repeat tensor n times along the first axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector (i.e., 1D tensor).<br />&nbsp;&nbsp;&nbsp;&nbsp;n: The number of repeats.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector created from many repeats of the input tensor.


### reshape
```py

def reshape(tensor: numpy.ndarray, newshape: Tuple[int]) -> numpy.ndarray

```



Return tensor with a new shape.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;newshape: The desired shape.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor with the desired shape.


### scatter\_
```py

def scatter_(mat: numpy.ndarray, inds: numpy.ndarray, val: Union[int, float]) -> numpy.ndarray

```



Assign a value a specific points in a matrix.<br />Iterates along the rows of mat,<br />successively assigning val to column indices given by inds.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies mat in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;inds: The indices<br />&nbsp;&nbsp;&nbsp;&nbsp;val: The value to insert


### set\_seed
```py

def set_seed(n: int = 137) -> None

```



Set the seed of the random number generator.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Default seed is 137.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n: Random seed.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### shape
```py

def shape(tensor: numpy.ndarray) -> Tuple[int]

```



Return a tuple with the shape of the tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tuple: A tuple of integers describing the shape of the tensor.


### shuffle\_
```py

def shuffle_(tensor: numpy.ndarray) -> None

```



Shuffle the rows of a tensor.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies tensor in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (shape): a tensor to shuffle.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### sign
```py

def sign(tensor: numpy.ndarray) -> numpy.ndarray

```



Return the elementwise sign of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The sign of the elements in the tensor.


### sin
```py

def sin(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise sine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise sine.


### softmax
```py

def softmax(x: numpy.ndarray, axis: int = 1) -> numpy.ndarray

```



Softmax function on a tensor.<br />Exponentiaties the tensor elementwise and divides<br />&nbsp;&nbsp;&nbsp;&nbsp;by the sum along axis=1.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Softmax of the tensor.


### softplus
```py

def softplus(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise softplus function of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise softplus.


### sort
```py

def sort(x: numpy.ndarray, axis: int = None) -> numpy.ndarray

```



Sort a tensor along the specied axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of floats): sorted tensor


### sqrt
```py

def sqrt(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square root of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (non-negative): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor(non-negative): Elementwise square root.


### sqrt\_div
```py

def sqrt_div(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise division of x by sqrt(y).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A non-negative tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise division of x by sqrt(y).


### square
```py

def square(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (non-negative): Elementwise square.


### square\_mix\_
```py

def square_mix_(w: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> None

```



Compute a weighted average of two matrices (x and y^2) and store the results in x.<br />Useful for keeping track of running averages of squared matrices during training.<br /><br />x <- w x + (1-w) * y**2<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies x in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;w: The mixing coefficient tensor between 0 and 1.<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### stack
```py

def stack(tensors: Iterable[numpy.ndarray], axis: int) -> numpy.ndarray

```



Stack tensors along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: A list of tensors.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis the tensors will be stacked along.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Stacked tensors from the input list.


### std
```py

def std(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the standard deviation of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the standard deviation.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall standard deviation of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The standard deviation of the tensor along the specified axis.


### subtract
```py

def subtract(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray

```



Subtract tensor a from tensor b using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: b - a


### subtract\_
```py

def subtract_(a: numpy.ndarray, b: numpy.ndarray) -> None

```



Subtract tensor a from tensor b using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### svd
```py

def svd(mat: numpy.ndarray) -> Tuple[numpy.ndarray]

```



Compute the Singular Value decomposition of a matrix<br />A = U S V^T<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;(U, S, V): Tuple of tensors.


### tabs
```py

def tabs(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise absolute value of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (non-negative): Absolute value of x.


### tall
```py

def tall(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[bool, numpy.ndarray]

```



Return True if all elements of the input tensor are true along the<br />specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis of interest.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bool: 'all' applied to all elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): 'all' applied to the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;along axis


### tanh
```py

def tanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic tangent of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise hyperbolic tangent.


### tany
```py

def tany(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[bool, numpy.ndarray]

```



Return True if any elements of the input tensor are true along the<br />specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis of interest.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bool: 'any' applied to all elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): 'any' applied to the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;along axis


### tceil
```py

def tceil(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor with ceilinged elements.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor rounded up to the next integer (still floating point).


### tclip
```py

def tclip(tensor: numpy.ndarray, a_min: numpy.ndarray = None, a_max: numpy.ndarray = None) -> numpy.ndarray

```



Return a tensor with its values clipped element-wise between a_min and a_max tensors.<br />The implementation is identical to clip.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional tensor): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional tensor): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A new tensor with its values clipped between a_min and a_max.


### tclip\_
```py

def tclip_(tensor: numpy.ndarray, a_min: numpy.ndarray = None, a_max: numpy.ndarray = None) -> None

```



Clip the values of a tensor elementwise between a_min and a_max tensors.<br />The implementation is identical to clip_<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies tensor in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional tensor): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional tessor): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### tfloor
```py

def tfloor(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor with floored elements.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor rounded down to the next integer (still floating point).


### tmax
```py

def tmax(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the elementwise maximum of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the maximum.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall maximum of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The maximum of the tensor along the specified axis.


### tmin
```py

def tmin(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the elementwise minimum of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the minimum.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall minimum of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The minimum of the tensor along the specified axis.


### tmul
```py

def tmul(a: Union[int, float], x: numpy.ndarray) -> numpy.ndarray

```



Elementwise multiplication of tensor x by scalar a.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a: scalar.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise a * x.


### tmul\_
```py

def tmul_(a: Union[int, float], x: numpy.ndarray)

```



Elementwise multiplication of tensor x by scalar a.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifes x in place<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a: scalar.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### to\_numpy\_array
```py

def to_numpy_array(tensor: numpy.ndarray) -> numpy.ndarray

```



Return tensor as a numpy array.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to a numpy array.


### tpow
```py

def tpow(x: numpy.ndarray, a: float) -> numpy.ndarray

```



Elementwise power of a tensor x to power a.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a: Power.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise x to the power of a.


### tprod
```py

def tprod(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the product of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the product.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall product of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The product of the tensor along the specified axis.


### trange
```py

def trange(start: int, end: int, step: int = 1, dtype: numpy.dtype = None) -> numpy.ndarray

```



Generate a tensor like a python range.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;start: The start of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;end: The end of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;step: The step of the range.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector ranging from start to end in increments<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of step. Cast to float rather than int.


### transpose
```py

def transpose(tensor: numpy.ndarray) -> numpy.ndarray

```



Return the transpose of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The transpose (exchange of rows and columns) of the tensor.


### tround
```py

def tround(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor with rounded elements.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor rounded to the nearest integer (still floating point).


### tsum
```py

def tsum(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the sum of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the sum.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall sum of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The sum of the tensor along the specified axis.


### unsqueeze
```py

def unsqueeze(tensor: numpy.ndarray, axis: int) -> numpy.ndarray

```



Return tensor with a new axis inserted.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The desired axis.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor with the new axis inserted.


### var
```py

def var(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[numpy.float32, numpy.ndarray]

```



Return the variance of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the variance.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall variance of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The variance of the tensor along the specified axis.


### vstack
```py

def vstack(tensors: Iterable[numpy.ndarray]) -> numpy.ndarray

```



Concatenate tensors along the zeroth axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: A list of tensors.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensors stacked along axis=0.


### zeros
```py

def zeros(shape: Tuple[int], dtype: numpy.dtype = <class 'numpy.float32'>) -> numpy.ndarray

```



Return a tensor of a specified shape filled with zeros.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: The shape of the desired tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of zeros with the desired shape.


### zeros\_like
```py

def zeros_like(tensor: numpy.ndarray) -> numpy.ndarray

```



Return a tensor of zeros with the same shape as the input tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of zeros with the same shape.

