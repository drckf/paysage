# Documentation for Matrix (matrix.py)

## class BroadcastError
BroadcastError exception:<br /><br />Args: None


## functions

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


### shape
```py

def shape(tensor: numpy.ndarray) -> Tuple[int]

```



Return a tuple with the shape of the tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tuple: A tuple of integers describing the shape of the tensor.


### sign
```py

def sign(tensor: numpy.ndarray) -> numpy.ndarray

```



Return the elementwise sign of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The sign of the elements in the tensor.


### sort
```py

def sort(x: numpy.ndarray, axis: int = None) -> numpy.ndarray

```



Sort a tensor along the specied axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of floats): sorted tensor


### sqrt\_div
```py

def sqrt_div(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray

```



Elementwise division of x by sqrt(y).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A non-negative tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise division of x by sqrt(y).


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


### tall
```py

def tall(x: numpy.ndarray, axis: int = None, keepdims: bool = False) -> Union[bool, numpy.ndarray]

```



Return True if all elements of the input tensor are true along the<br />specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis of interest.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bool: 'all' applied to all elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): 'all' applied to the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;along axis


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


### to\_numpy\_array
```py

def to_numpy_array(tensor: numpy.ndarray) -> numpy.ndarray

```



Return tensor as a numpy array.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to a numpy array.


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

