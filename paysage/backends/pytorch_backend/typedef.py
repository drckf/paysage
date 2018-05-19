from typing import Iterable, Tuple, Union, Dict, List
from numpy import ndarray
import numpy
import torch
from torch import IntTensor as IntTensorCPU
from torch import ShortTensor as ShortTensorCPU
from torch import LongTensor as LongTensorCPU
from torch import ByteTensor as ByteTensorCPU
from torch import FloatTensor as FloatTensorCPU
from torch import DoubleTensor as DoubleTensorCPU
from torch.cuda import IntTensor as IntTensorGPU
from torch.cuda import ShortTensor as ShortTensorGPU
from torch.cuda import LongTensor as LongTensorGPU
from torch.cuda import ByteTensor as ByteTensorGPU
from torch.cuda import FloatTensor as FloatTensorGPU
from torch.cuda import DoubleTensor as DoubleTensorGPU

IntTensor = Union[IntTensorCPU, IntTensorGPU]
ShortTensor = Union[ShortTensorCPU, ShortTensorGPU]
LongTensor = Union[LongTensorCPU, LongTensorGPU]
ByteTensor = Union[ByteTensorCPU, ByteTensorGPU]
FloatTensor = Union[FloatTensorCPU, FloatTensorGPU]
DoubleTensor = Union[DoubleTensorCPU, DoubleTensorGPU]

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

FloatConstructable = Union[FloatTensor,
                           NumpyTensor,
                           Iterable[float]]

LongConstructable = Union[LongTensor,
                          NumpyTensor,
                          Iterable[int]]

Int = torch.int32
Long = torch.int64
Float = torch.float32
Double = torch.float64
Byte = torch.uint8

Dtype = torch.dtype
torch.set_default_dtype(torch.float32)

EPSILON = float(numpy.finfo(numpy.float32).eps)
