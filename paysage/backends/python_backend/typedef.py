from typing import Iterable, Tuple, Union, Dict
from numpy import ndarray
from numpy import float32

Scalar = Union[int, float]

FloatingPoint = Union[float32, ndarray]
# assuming here that ndarray.dtype == float32
# no good way to specify this right now

Boolean = Union[bool, ndarray]
# assuming here that ndarray.dtype == bool
# no good way to specify this right now

Tensor = ndarray

NumpyTensor = ndarray

Numeric = Union[Scalar, Tensor]
