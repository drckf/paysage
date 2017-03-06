#from .python_backend import matrix as pymat
#from .pytorch_backend import matrix as torchmat

import python_backend.matrix as py_matrix
import python_backend.nonlinearity as py_func
import python_backend.rand as py_rand

import pytorch_backend.matrix as torch_matrix
import pytorch_backend.nonlinearity as torch_func
import pytorch_backend.rand as torch_rand

import pytest

def test_conversion():

    shape = (100, 100)

    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)
    py_torch_x = torch_matrix.to_numpy_array(torch_x)

    assert py_matrix.allclose(py_x, py_torch_x), \
    "python -> torch -> python failure"

    torch_y = torch_rand.rand(shape)
    py_y = torch_matrix.to_numpy_array(torch_y)
    torch_py_y = torch_matrix.float_tensor(py_y)

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure"


if __name__ == "__main__":
    test_conversion()