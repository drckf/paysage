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

def test_acosh():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = 1 + py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.acosh(py_x)
    torch_y = torch_func.acosh(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: acosh"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: acosh"

def test_tabs():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.tabs(py_x)
    torch_y = torch_func.tabs(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: tabs"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: tabs"

def test_exp():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.exp(py_x)
    torch_y = torch_func.exp(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: exp"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: exp"

def test_log():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.log(py_x)
    torch_y = torch_func.log(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: exp"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: exp"



if __name__ == "__main__":
    test_conversion()
    test_acosh()
    test_exp()
    test_log()
    test_tabs()
