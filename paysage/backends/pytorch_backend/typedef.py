from typing import Iterable, Tuple, Union, Dict
from numpy import ndarray
from torch import IntTensor, ShortTensor, LongTensor
from torch import ByteTensor
from torch import FloatTensor, DoubleTensor

Scalar = Union[int, float]

FloatingPoint = Union[float, FloatTensor]

Boolean = Union[bool, ByteTensor]

NumpyTensor = ndarray

TorchTensor = Union[IntTensor,
               ShortTensor,
               LongTensor,
               ByteTensor,
               FloatTensor,
               DoubleTensor]

Tensor = Union[NumpyTensor, TorchTensor]

Numeric = Union[Scalar, Tensor]
