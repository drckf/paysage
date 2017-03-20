from typing import Iterable, Tuple, Union, Dict
from numpy import ndarray
from numpy import float32

Scalar = Union[int, float]

FloatingPoint = Union[float32, ndarray]
# aasuming here that ndarray.dtype == float32
# no good way to specify this right now

Tensor = ndarray
Numeric = Union[Scalar, Tensor]
