# Documentation for Matrix (matrix.py)

## class BroadcastError
BroadcastError exception:<br /><br />Args: None


## functions

### add
```py

def add(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Add tensor a to tensor b using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: a + b


### add\_
```py

def add_(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None

```



Add tensor a to tensor b using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### affine
```py

def affine(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor], W: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Evaluate the affine transformation a + W b.<br /><br />a ~ vector, b ~ vector, W ~ matrix:<br />a_i + \sum_j W_ij b_j<br /><br />a ~ matrix, b ~ matrix, W ~ matrix:<br />a_ij + \sum_k W_ik b_kj<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor (1 or 2 dimensional).<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor (1 or 2 dimensional).<br />&nbsp;&nbsp;&nbsp;&nbsp;W: A tensor (2 dimensional).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Affine transformation a + W b.


### allclose
```py

def allclose(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor], rtol: float=1e-05, atol: float=1e-08) -> bool

```



Test if all elements in the two tensors are approximately equal.<br /><br />absolute(x - y) <= (atol + rtol * absolute(y))<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;rtol (optional): Relative tolerance.<br />&nbsp;&nbsp;&nbsp;&nbsp;atol (optional): Absolute tolerance.<br /><br />returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: Check if all of the elements in the tensors are approximately equal.


### argmax
```py

def argmax(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int) -> Union[torch.LongTensor, torch.cuda.LongTensor]

```



Compute the indices of the maximal elements in x along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of ints): Indices of the maximal elements in x along the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  specified axis.


### argmin
```py

def argmin(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None) -> Union[torch.LongTensor, torch.cuda.LongTensor]

```



Compute the indices of the minimal elements in x along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of ints): Indices of the minimum elements in x along the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  specified axis.


### argsort
```py

def argsort(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None) -> Union[torch.LongTensor, torch.cuda.LongTensor]

```



Get the indices of a sorted tensor.<br />If axis=None this flattens x.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of ints): indices of sorted tensor


### batch\_dot
```py

def batch_dot(a: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], b: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], axis: int=1) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Compute the dot product of vectors batch-wise.<br />Let a be an L x N matrix where each row a_i is a vector.<br />Let b be an L x N matrix where each row b_i is a vector.<br />Then batch_dot(a, b) = \sum_j a_ij * b_ij<br />One can choose the axis to sum along with the axis argument.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (int): The axis to dot along.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.


### batch\_outer
```py

def batch_outer(vis: Union[torch.FloatTensor, torch.cuda.FloatTensor], hid: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Let v by a L x N matrix where each row v_i is a visible vector.<br />Let h be a L x M matrix where each row h_i is a hidden vector.<br />Then, batch_outer(v, h) = \sum_i v_i h_i^T<br />Returns an N x M matrix.<br /><br />The actual computation is performed with a vectorized expression.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vis: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;hid: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A matrix.


### batch\_quadratic
```py

def batch_quadratic(vis: Union[torch.FloatTensor, torch.cuda.FloatTensor], W: Union[torch.FloatTensor, torch.cuda.FloatTensor], hid: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=1) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Let v by a L x N matrix where each row v_i is a visible vector.<br />Let h be a L x M matrix where each row h_i is a hidden vector.<br />And, let W be a N x M matrix of weights.<br />Then, batch_quadratic(v,W,h) = \sum_i v_i^T W h_i<br /><br />The actual computation is performed with a vectorized expression.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vis: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;W: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;hid: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): Axis of interest<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector.


### broadcast
```py

def broadcast(vec: Union[torch.FloatTensor, torch.cuda.FloatTensor], matrix: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Broadcasts vec into the shape of matrix following numpy rules:<br /><br />vec ~ (N, 1) broadcasts to matrix ~ (N, M)<br />vec ~ (1, N) and (N,) broadcast to matrix ~ (M, N)<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vec: A vector (either flat, row, or column).<br />&nbsp;&nbsp;&nbsp;&nbsp;matrix: A matrix (i.e., a 2D tensor).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of the same size as matrix containing the elements<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of the vector.<br /><br />Raises:<br />&nbsp;&nbsp;&nbsp;&nbsp;BroadcastError


### cast\_float
```py

def cast_float(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Cast torch tensor to a float tensor.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;If tensor is already float, no copy is made.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to floating point.


### cast\_long
```py

def cast_long(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[torch.LongTensor, torch.cuda.LongTensor]

```



Cast torch tensor to a long int tensor.<br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;If tensor is already long int, no copy is made.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A torch tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to long.


### center
```py

def center(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=0) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Remove the mean along axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units): the array to center<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (int; optional): the axis to center along<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units)


### clip
```py

def clip(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor], a_min: Union[int, float]=-inf, a_max: Union[int, float]=inf) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return a tensor with its values clipped between a_min and a_max.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A new tensor with its values clipped between a_min and a_max.


### clip\_
```py

def clip_(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor], a_min: Union[int, float]=-inf, a_max: Union[int, float]=inf) -> None

```



Clip the values of a tensor between a_min and a_max.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies tensor in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### copy\_tensor
```py

def copy_tensor(tensor: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Copy a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;copy of tensor


### corr
```py

def corr(x: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], y: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Compute the cross correlation between tensors x and y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor (num_samples, num_units_x))<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor (num_samples, num_units_y))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_units_x, num_units_y)


### cov
```py

def cov(x: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], y: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Compute the cross covariance between tensors x and y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor (num_samples, num_units_x))<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor (num_samples, num_units_y))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_units_x, num_units_y)


### cumsum
```py

def cumsum(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=0) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the cumulative sum of elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the sum.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the cumulative sum of elements of the tensor along the specified axis.


### diag
```py

def diag(mat: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return the diagonal elements of a matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector (i.e., 1D tensor) containing the diagonal<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;elements of mat.


### diagonal\_matrix
```py

def diagonal_matrix(vec: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return a matrix with vec along the diagonal.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vec: A vector (i.e., 1D tensor).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A matrix with the elements of vec along the diagonal,<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and zeros elsewhere.


### divide
```py

def divide(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Divide tensor b by tensor a using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor (non-zero)<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: b / a


### divide\_
```py

def divide_(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None

```



Divide tensor b by tensor a using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor (non-zero)<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### dot
```py

def dot(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Compute the matrix/dot product of tensors a and b.<br /><br />Vector-Vector:<br />&nbsp;&nbsp;&nbsp;&nbsp;\sum_i a_i b_i<br /><br />Matrix-Vector:<br />&nbsp;&nbsp;&nbsp;&nbsp;\sum_j a_ij b_j<br /><br />Matrix-Matrix:<br />&nbsp;&nbsp;&nbsp;&nbsp;\sum_j a_ij b_jk<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if a and b are 1-dimensions:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: the dot product of vectors a and b<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: the matrix product of tensors a and b


### dtype
```py

def dtype(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> type

```



Return the type of the tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;type: The type of the elements in the tensor.


### equal
```py

def equal(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Elementwise test for if two tensors are equal.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of equality between x and y.


### fill\_diagonal\_
```py

def fill_diagonal_(mat: Union[torch.FloatTensor, torch.cuda.FloatTensor], val: Union[int, float]) -> None

```



Fill the diagonal of the matirx with a specified value.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies mat in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;val: The value to put along the diagonal.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### flatten
```py

def flatten(tensor: Union[float, torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return a flattened tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor or scalar.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;result: If arg is a tensor, return a flattened 1D tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If arg is a scalar, return the scalar.


### float\_scalar
```py

def float_scalar(scalar: Union[int, float]) -> float

```



Cast scalar to a float.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;scalar: A scalar quantity:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float: Scalar converted to floating point.


### float\_tensor
```py

def float_tensor(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor, numpy.ndarray, Iterable[float]]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Construct to a float tensor.  Note: requires floating point data.<br />This will always copy the data in tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to floating point.


### from\_numpy\_array
```py

def from_numpy_array(tensor: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> numpy.ndarray

```



Construct a tensor from a numpy array.<br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;This shares the memory with the ndarray.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A numpy ndarray<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted from ndarray.


### greater
```py

def greater(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Elementwise test if x > y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x > y.


### greater\_equal
```py

def greater_equal(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Elementwise test if x >= y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x >= y.


### hstack
```py

def hstack(tensors: Iterable[Union[torch.FloatTensor, torch.cuda.FloatTensor]]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Concatenate tensors along the first axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: A list of tensors.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensors stacked along axis=1.


### identity
```py

def identity(n: int) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the n-dimensional identity matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n: The desired size of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The n x n identity matrix with ones along the diagonal<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and zeros elsewhere.


### index\_select
```py

def index_select(mat: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], index: Union[torch.LongTensor, torch.cuda.LongTensor], dim: int=0) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Select the specified indices of a tensor along dimension dim.<br />For example, dim = 1 is equivalent to mat[:, index] in numpy.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat (tensor (num_samples, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;index (tensor; 1 -dimensional)<br />&nbsp;&nbsp;&nbsp;&nbsp;dim (int)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if dim == 0:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mat[index, :]<br />&nbsp;&nbsp;&nbsp;&nbsp;if dim == 1:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mat[:, index]


### inv
```py

def inv(mat: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Compute matrix inverse.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The matrix inverse.


### is\_tensor
```py

def is_tensor(x: Union[float, torch.FloatTensor, torch.cuda.FloatTensor]) -> bool

```



Test if x is a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (float or tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool


### lesser
```py

def lesser(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Elementwise test if x < y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x < y.


### lesser\_equal
```py

def lesser_equal(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Elementwise test if x <= y.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bools): Elementwise test of x <= y.


### logdet
```py

def logdet(mat: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> float

```



Compute the logarithm of the determinant of a square matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;logdet: The logarithm of the matrix determinant.


### logical\_and
```py

def logical_and(x: Union[torch.ByteTensor, torch.cuda.ByteTensor], y: Union[torch.ByteTensor, torch.cuda.ByteTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Compute the elementwise logical and on two tensors<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### logical\_not
```py

def logical_not(x: Union[torch.ByteTensor, torch.cuda.ByteTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Invert a logical array (True -> False, False -> True).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### logical\_or
```py

def logical_or(x: Union[torch.ByteTensor, torch.cuda.ByteTensor], y: Union[torch.ByteTensor, torch.cuda.ByteTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Compute the elementwise logical or on two tensors<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br />&nbsp;&nbsp;&nbsp;&nbsp;y (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### long\_tensor
```py

def long_tensor(tensor: Union[torch.LongTensor, torch.cuda.LongTensor, numpy.ndarray, Iterable[int]]) -> Union[torch.LongTensor, torch.cuda.LongTensor]

```



Construct a long tensor.  This will always copy the data in tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to long int type.


### matrix\_sqrt
```py

def matrix_sqrt(mat: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Compute the matrix square root using an SVD<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;matrix square root


### maximum
```py

def maximum(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Elementwise maximum of two tensors.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise maximum of x and y.


### mean
```py

def mean(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the mean of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor of rank=2.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the mean.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall mean of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The mean of the tensor along the specified axis.


### minimum
```py

def minimum(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Elementwise minimum of two tensors.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise minimum of x and y.


### mix
```py

def mix(w: Union[float, torch.FloatTensor, torch.cuda.FloatTensor], x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None

```



Compute a weighted average of two matrices (x and y) and return the result.<br />Multilinear interpolation.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies x in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;w: The mixing coefficient (float or tensor) between 0 and 1.<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor = w * x + (1-w) * y


### mix\_
```py

def mix_(w: Union[float, torch.FloatTensor, torch.cuda.FloatTensor], x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None

```



Compute a weighted average of two matrices (x and y) and store the results in x.<br />Useful for keeping track of running averages during training.<br /><br />x <- w * x + (1-w) * y<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies x in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;w: The mixing coefficient (float or tensor) between 0 and 1.<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### multiply
```py

def multiply(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Multiply tensor b with tensor a using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: a * b


### multiply\_
```py

def multiply_(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None

```



Multiply tensor b with tensor a using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### ndim
```py

def ndim(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> int

```



Return the number of dimensions of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;int: The number of dimensions of the tensor.


### norm
```py

def norm(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the L2 norm of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): the axis for taking the norm<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is none:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The L2 norm of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   (i.e., the sqrt of the sum of the squared elements).<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The L2 norm along the specified axis.


### normalize
```py

def normalize(x: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Divide x by it's sum.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A non-negative tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor normalized by it's sum.


### not\_equal
```py

def not_equal(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.ByteTensor, torch.cuda.ByteTensor]

```



Elementwise test if two tensors are not equal.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bytes): Elementwise test of non-equality between x and y.


### num\_elements
```py

def num_elements(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> int

```



Return the number of elements in a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;int: The number of elements in the tensor.


### ones
```py

def ones(shape: Tuple[int], dtype: torch.dtype=torch.float32) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return a tensor of a specified shape filled with ones.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: The shape of the desired tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of ones with the desired shape.


### ones\_like
```py

def ones_like(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return a tensor of ones with the same shape and dtype as the input tensor.<br />Note: much faster on the GPU than calling ones(shape(tensor))<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor with the same shape.


### outer
```py

def outer(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Compute the outer product of vectors x and y.<br /><br />mat_{ij} = x_i * y_j<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A vector (i.e., a 1D tensor).<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A vector (i.e., a 1D tensor).<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Outer product of vectors x and y.


### pinv
```py

def pinv(mat: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Compute matrix pseudoinverse.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A square matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The matrix pseudoinverse.


### qr
```py

def qr(mat: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]

```



Compute the QR decomposition of a matrix.<br />The QR decomposition factorizes a matrix A into a product<br />A = QR of an orthonormal matrix Q and an upper triangular matrix R.<br />Provides an orthonormalization of the columns of the matrix.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;(Q, R): Tuple of tensors.


### quadratic
```py

def quadratic(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor], W: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Evaluate the quadratic form a W b.<br /><br />a ~ vector, b ~ vector, W ~ matrix:<br />\sum_ij a_i W_ij b_j<br /><br />a ~ matrix, b ~ matrix, W ~ matrix:<br />\sum_kl a_ik W_kl b_lj<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;W: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Quadratic function a W b.


### repeat
```py

def repeat(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor], n: int) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Repeat tensor n times along specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector (i.e., 1D tensor).<br />&nbsp;&nbsp;&nbsp;&nbsp;n: The number of repeats.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector created from many repeats of the input tensor.


### reshape
```py

def reshape(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor], newshape: Tuple[int]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return tensor with a new shape.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;newshape: The desired shape.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor with the desired shape.


### scatter\_
```py

def scatter_(mat: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], inds: Union[torch.LongTensor, torch.cuda.LongTensor], val: Union[int, float]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Assign a value a specific points in a matrix.<br />Iterates along the rows of mat,<br />successively assigning val to column indices given by inds.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies mat in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;inds: The indices<br />&nbsp;&nbsp;&nbsp;&nbsp;val: The value to insert


### shape
```py

def shape(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Tuple[int]

```



Return a tuple with the shape of the tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tuple: A tuple of integers describing the shape of the tensor.


### sign
```py

def sign(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the elementwise sign of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The sign of the elements in the tensor.


### sort
```py

def sort(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Sort a tensor along the specied axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis of interest.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (of floats): sorted tensor


### sqrt\_div
```py

def sqrt_div(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Elementwise division of x by sqrt(y).<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor:<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A non-negative tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise division of x by sqrt(y).


### square\_mix\_
```py

def square_mix_(w: Union[float, torch.FloatTensor, torch.cuda.FloatTensor], x: Union[torch.FloatTensor, torch.cuda.FloatTensor], y: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None

```



Compute a weighted average of two matrices (x and y^2) and store the results in x.<br />Useful for keeping track of running averages of squared matrices during training.<br /><br />x <- w x + (1-w) * y**2<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies x in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;w: The mixing coefficient (float or tensor) between 0 and 1 .<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;y: A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### stack
```py

def stack(tensors: Iterable[Union[torch.FloatTensor, torch.cuda.FloatTensor]], axis: int) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Stack tensors along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: A list of tensors.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The axis the tensors will be stacked along.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Stacked tensors from the input list.


### std
```py

def std(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the standard deviation of the elements of a tensor along the<br />specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the standard deviation.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall standard deviation of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The standard deviation of the tensor along the specified axis.


### subtract
```py

def subtract(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Subtract tensor a from tensor b using broadcasting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: b - a


### subtract\_
```py

def subtract_(a: Union[torch.FloatTensor, torch.cuda.FloatTensor], b: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> None

```



Subtract tensor a from tensor b using broadcasting.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies b in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;a: A tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;b: A tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### svd
```py

def svd(mat: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Tuple[Union[torch.FloatTensor, torch.cuda.FloatTensor]]

```



Compute the Singular Value decomposition of a matrix<br />A = U S V^T<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mat: A matrix.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;(U, S, V): Tuple of tensors.


### tall
```py

def tall(x: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], axis: int=None, keepdims: bool=False) -> Union[bool, torch.ByteTensor, torch.cuda.ByteTensor]

```



Return True if all elements of the input tensor are true along the<br />specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis of interest.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bool: 'all' applied to all elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bytes): 'all' applied to the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;along axis


### tany
```py

def tany(x: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], axis: int=None, keepdims: bool=False) -> Union[bool, torch.ByteTensor, torch.cuda.ByteTensor]

```



Return True if any elements of the input tensor are true along the<br />specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: (torch.ByteTensor)<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis of interest.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bool: 'any' applied to all elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor (of bytes): 'any' applied to the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;along axis


### tceil
```py

def tceil(tensor: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return a tensor with ceilinged elements.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor rounded up to the next integer (still floating point).


### tclip
```py

def tclip(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor], a_min: Union[torch.FloatTensor, torch.cuda.FloatTensor]=None, a_max: Union[torch.FloatTensor, torch.cuda.FloatTensor]=None) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return a tensor with its values clipped element-wise between a_min and a_max tensors.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A new tensor with its values clipped between a_min and a_max.


### tclip\_
```py

def tclip_(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor], a_min: Union[torch.FloatTensor, torch.cuda.FloatTensor]=None, a_max: Union[torch.FloatTensor, torch.cuda.FloatTensor]=None) -> None

```



Clip the values of a tensor elementwise between a_min and a_max tensors.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies tensor in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (optional): The desired lower bound on the elements of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (optional): The desired upper bound on the elements of the tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### tfloor
```py

def tfloor(tensor: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return a tensor with floored elements.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor rounded down to the next integer (still floating point).


### tmax
```py

def tmax(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the elementwise maximum of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the maximum.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall maximum of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The maximum of the tensor along the specified axis.


### tmin
```py

def tmin(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the elementwise minimum of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the minimum.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall minimum of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The minimum of the tensor along the specified axis.


### to\_numpy\_array
```py

def to_numpy_array(tensor: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> numpy.ndarray

```



Return tensor as a numpy array.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensor converted to a numpy array.


### tprod
```py

def tprod(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the product of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the product.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall product of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The product of the tensor along the specified axis.


### trange
```py

def trange(start: int, end: int, step: int=1, dtype=torch.float32) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Generate a tensor like a python range.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;start: The start of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;end: The end of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;step: The step of the range.<br />&nbsp;&nbsp;&nbsp;&nbsp;dtype: (torch.Dtype): desired data type for output<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A vector ranging from start to end in increments<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of step. Cast to float rather than int.


### transpose
```py

def transpose(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the transpose of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: The transpose (exchange of rows and columns) of the tensor.


### tround
```py

def tround(tensor: Union[torch.FloatTensor, torch.cuda.FloatTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return a tensor with rounded elements.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor rounded to the nearest integer (still floating point).


### tsum
```py

def tsum(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the sum of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the sum.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall sum of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The sum of the tensor along the specified axis.


### unsqueeze
```py

def unsqueeze(tensor: Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor], axis: int) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return tensor with a new axis inserted.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis: The desired axis.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor with the new axis inserted.


### var
```py

def var(x: Union[torch.FloatTensor, torch.cuda.FloatTensor], axis: int=None, keepdims: bool=False) -> Union[float, torch.FloatTensor, torch.cuda.FloatTensor]

```



Return the variance of the elements of a tensor along the specified axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A float or tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;axis (optional): The axis for taking the variance.<br />&nbsp;&nbsp;&nbsp;&nbsp;keepdims (optional): If this is set to true, the dimension of the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; is unchanged. Otherwise, the reduced axis is removed<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; and the dimension of the array is 1 less.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;if axis is None:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;float: The overall variance of the elements in the tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;else:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor: The variance of the tensor along the specified axis.


### vstack
```py

def vstack(tensors: Iterable[Union[torch.FloatTensor, torch.cuda.FloatTensor]]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Concatenate tensors along the zeroth axis.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: A list of tensors.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Tensors stacked along axis=0.


### zeros
```py

def zeros(shape: Tuple[int], dtype: torch.dtype=torch.float32) -> Union[numpy.ndarray, torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]

```



Return a tensor of a specified shape filled with zeros.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape: The shape of the desired tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of zeros with the desired shape.


### zeros\_like
```py

def zeros_like(tensor: Union[torch.IntTensor, torch.cuda.IntTensor, torch.ShortTensor, torch.cuda.ShortTensor, torch.LongTensor, torch.cuda.LongTensor, torch.ByteTensor, torch.cuda.ByteTensor, torch.FloatTensor, torch.cuda.FloatTensor, torch.DoubleTensor, torch.cuda.DoubleTensor]) -> Union[torch.FloatTensor, torch.cuda.FloatTensor]

```



Return a tensor of zeros with the same shape and dtype as the input tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: A tensor of zeros with the same shape.

