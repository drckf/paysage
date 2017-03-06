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

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)
    py_torch_x = torch_matrix.to_numpy_array(torch_x)

    assert py_matrix.allclose(py_x, py_torch_x), \
    "python -> torch -> python failure"

    torch_rand.set_seed()
    torch_y = torch_rand.rand(shape)
    py_y = torch_matrix.to_numpy_array(torch_y)
    torch_py_y = torch_matrix.float_tensor(py_y)

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure"

def test_nonlinearities():
    from collections import OrderedDict
    from inspect import getmembers, isfunction
    python_nonlinearities = OrderedDict(getmembers(py_func, isfunction))
    torch_nonlinearities = OrderedDict(getmembers(torch_func, isfunction))

    shape = (100, 100)
    py_rand.set_seed()
    py_x = 1 + py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    for key in python_nonlinearities:
        print(key)
        py_y = python_nonlinearities[key](py_x)
        torch_y = torch_nonlinearities[key](torch_x)

        torch_py_y = torch_matrix.float_tensor(py_y)
        py_torch_y = torch_matrix.to_numpy_array(torch_y)

        assert py_matrix.allclose(py_y, py_torch_y), \
        "python -> torch -> python failure: {}".format(key)

        assert torch_matrix.allclose(torch_y, torch_py_y), \
        "torch -> python -> torch failure: {}".format(key)


if __name__ == "__main__":
    test_conversion()
    test_nonlinearities()
