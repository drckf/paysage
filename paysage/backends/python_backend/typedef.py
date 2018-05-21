from typing import Iterable, Tuple, Union, Dict
from numpy import ndarray
import numpy

Scalar = Union[int, float]

FloatingPoint = Union[numpy.float32, ndarray]
# assuming here that ndarray.dtype == float32
# no good way to specify this right now

Boolean = Union[bool, ndarray]
# assuming here that ndarray.dtype == bool
# no good way to specify this right now

Tensor = ndarray

NumpyTensor = ndarray

Numeric = Union[Scalar, Tensor]

FloatConstructable = Union[Tensor,
                           Iterable[float]]

LongConstructable = Union[Tensor,
                          Iterable[int]]

Int = numpy.int32
Long = numpy.int64
Float = numpy.float32
Double = numpy.float64
Byte = numpy.uint8

Dtype = numpy.dtype

EPSILON = float(numpy.finfo(Float).eps)
