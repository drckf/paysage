import paysage.backends.python_backend.matrix as py_matrix
import paysage.backends.python_backend.nonlinearity as py_func
import paysage.backends.python_backend.rand as py_rand

import paysage.backends.pytorch_backend.matrix as torch_matrix
import paysage.backends.pytorch_backend.nonlinearity as torch_func
import paysage.backends.pytorch_backend.rand as torch_rand

from numpy import allclose
import pytest

# ---- testing utility functions ----- #

def assert_close(pymat, torchmat, name):

    pytorchmat = torch_matrix.to_numpy_array(torchmat)
    torchpymat = torch_matrix.float_tensor(pymat)

    py_vs_torch = py_matrix.allclose(pymat, pytorchmat)
    torch_vs_py = torch_matrix.allclose(torchmat, torchpymat)

    if py_vs_torch and torch_vs_py:
        return
    if py_vs_torch and not torch_vs_py:
        assert False,\
        "{}: failure at torch allclose".format(name)
    elif not py_vs_torch and torch_vs_py:
        assert False, \
        "{}: failure at python allclose".format(name)
    else:
        assert False, \
        "{}: failure at both python and torch allclose".format(name)


# ----- Tests ------ #


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

def test_transpose():

    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_x_T = py_matrix.transpose(py_x)
    py_torch_x_T = torch_matrix.to_numpy_array(torch_matrix.transpose(torch_x))

    assert py_matrix.allclose(py_x_T, py_torch_x_T), \
    "python -> torch -> python failure: transpose"

    torch_rand.set_seed()
    torch_y = torch_rand.rand(shape)
    py_y = torch_matrix.to_numpy_array(torch_y)

    torch_y_T = torch_matrix.transpose(torch_y)
    torch_py_y_T = torch_matrix.float_tensor(py_matrix.transpose(py_y))

    assert torch_matrix.allclose(torch_y_T, torch_py_y_T), \
    "torch -> python -> torch failure: transpose"

def test_zeros():
    shape = (100, 100)

    py_zeros = py_matrix.zeros(shape)
    torch_zeros = torch_matrix.zeros(shape)
    assert_close(py_zeros, torch_zeros, "zeros")

def test_ones():
    shape = (100, 100)

    py_ones = py_matrix.ones(shape)
    torch_ones = torch_matrix.ones(shape)
    assert_close(py_ones, torch_ones, "ones")

def test_diag():
    shape = (100,)

    py_rand.set_seed()
    py_vec = py_rand.randn(shape)
    py_mat = py_matrix.diagonal_matrix(py_vec)
    py_diag = py_matrix.diag(py_mat)

    assert py_matrix.allclose(py_vec, py_diag), \
    "python vec -> matrix -> vec failure: diag"

    torch_vec = torch_rand.randn(shape)
    torch_mat = torch_matrix.diagonal_matrix(torch_vec)
    torch_diag = torch_matrix.diag(torch_mat)

    assert torch_matrix.allclose(torch_vec, torch_diag), \
    "torch vec -> matrix -> vec failure: diag"

def test_fill_diagonal():

    n = 10

    py_mat = py_matrix.identity(n)
    torch_mat = torch_matrix.identity(n)

    fill_value = 2.0

    py_mult = fill_value * py_mat
    py_matrix.fill_diagonal(py_mat, fill_value)

    assert py_matrix.allclose(py_mat, py_mult), \
    "python fill != python multiplly for diagonal matrix"

    torch_mult = fill_value * torch_mat
    torch_matrix.fill_diagonal(torch_mat, fill_value)

    assert torch_matrix.allclose(torch_mat, torch_mult), \
    "torch fill != python multiplly for diagonal matrix"

    assert_close(py_mat, torch_mat, "fill_diagonal")

def test_sign():

    shape = (100,100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_sign = py_matrix.sign(py_mat)
    torch_sign = torch_matrix.sign(torch_mat)
    assert_close(py_sign, torch_sign, "sign")

def test_clip():

    shape = (100,100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # test two sided clip
    py_clipped = py_matrix.clip(py_mat, a_min=0, a_max=1)
    torch_clipped = torch_matrix.clip(torch_mat, a_min=0, a_max=1)
    assert_close(py_clipped, torch_clipped, "clip (two-sided)")

    # test lower clip
    py_clipped = py_matrix.clip(py_mat, a_min=0)
    torch_clipped = torch_matrix.clip(torch_mat, a_min=0)
    assert_close(py_clipped, torch_clipped, "clip (lower)")

    # test upper clip
    py_clipped = py_matrix.clip(py_mat, a_max=1)
    torch_clipped = torch_matrix.clip(torch_mat, a_max=1)
    assert_close(py_clipped, torch_clipped, "clip (upper)")

def test_clip_inplace():

    shape = (100,100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # test two sided clip
    py_matrix.clip_inplace(py_mat, a_min=0, a_max=1)
    torch_matrix.clip_inplace(torch_mat, a_min=0, a_max=1)

    assert_close(py_mat, torch_mat, "clip_inplace (two-sided)")

    # test lower clip
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_matrix.clip_inplace(py_mat, a_min=0)
    torch_matrix.clip_inplace(torch_mat, a_min=0)

    assert_close(py_mat, torch_mat, "clip_inplace (lower)")

    # test upper clip
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_matrix.clip_inplace(py_mat, a_max=1)
    torch_matrix.clip_inplace(torch_mat, a_max=1)

    assert_close(py_mat, torch_mat, "clip_inplace (upper)")

def test_tround():

    shape = (100,100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_round = py_matrix.tround(py_mat)
    torch_round = torch_matrix.tround(torch_mat)

    assert_close(py_round, torch_round, "tround")

def test_flatten():
    # flatten a scalar
    # in contrast to numpy (which returns a 1 element array)
    # the backend flatten functions do nothing to scalars
    scalar = 5.7
    py_scalar = py_matrix.flatten(scalar)
    torch_scalar = torch_matrix.flatten(scalar)

    assert py_scalar == torch_scalar, \
    "error applying flatten to a scalar"

    # flatten a tensor
    shape = (100,100)
    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_flatten = py_matrix.flatten(py_mat)
    torch_flatten = torch_matrix.flatten(torch_mat)

    assert_close(py_flatten, torch_flatten, "flatten")

def test_reshape():
    shape = (100,100)
    newshape = (5, 2000)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_new = py_matrix.reshape(py_mat, newshape)
    torch_new = torch_matrix.reshape(torch_mat, newshape)

    assert_close(py_new, torch_new, "reshape")

def test_mix_inplace():
    shape = (100,100)
    torch_w = 0.1
    py_w = py_matrix.float_scalar(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.mix_inplace(py_w, py_x, py_y)
    torch_matrix.mix_inplace(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "mix_inplace")

def test_square_mix_inplace():
    shape = (100,100)
    torch_w = 0.1
    py_w = py_matrix.float_scalar(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.square_mix_inplace(py_w, py_x, py_y)
    torch_matrix.square_mix_inplace(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "square_mix_inplace")

def test_sqrt_div():
    shape = (100,100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape) ** 2

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_sqrt_div = py_matrix.sqrt_div(py_x, py_y)
    torch_sqrt_div = torch_matrix.sqrt_div(torch_x, torch_y)

    assert_close(py_sqrt_div, torch_sqrt_div, "sqrt_div")

def test_normalize():
    shape = (100,)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)

    torch_x = torch_matrix.float_tensor(py_x)

    py_norm = py_matrix.normalize(py_x)
    torch_norm = torch_matrix.normalize(torch_x)

    assert_close(py_norm, torch_norm, "normalize")

def test_norm():
    shape = (100,)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_norm = py_matrix.norm(py_x)
    torch_norm = torch_matrix.norm(torch_x)

    assert allclose(py_norm, torch_norm), \
    "python l2 norm != torch l2 norm"

def test_tmax():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # overall max
    py_max = py_matrix.tmax(py_mat)
    torch_max = torch_matrix.tmax(torch_mat)

    assert allclose(py_max, torch_max), \
    "python overal max != torch overall max"

    # max over axis 0
    py_max = py_matrix.tmax(py_mat, axis=0)
    torch_max = torch_matrix.tmax(torch_mat, axis=0)
    assert_close(py_max, torch_max, "allclose (axis-0)")

    # max over axis 1
    py_max = py_matrix.tmax(py_mat, axis=1)
    torch_max = torch_matrix.tmax(torch_mat, axis=1)
    assert_close(py_max, torch_max, "allclose (axis-1)")

    # max over axis 0, keepdims = True
    py_max = py_matrix.tmax(py_mat, axis=0, keepdims=True)
    torch_max = torch_matrix.tmax(torch_mat, axis=0)
    assert_close(py_max, torch_max, "allclose (axis-0, keepdims)")

    # max over axis 1, keepdims = True
    py_max = py_matrix.tmax(py_mat, axis=1, keepdims=True)
    torch_max = torch_matrix.tmax(torch_mat, axis=1, keepdims=True)
    assert_close(py_max, torch_max, "allclose (axis-1, keepdims)")


# ----- Nonlinearities ----- #

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
    "python -> torch -> python failure: log"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: log"

def test_tanh():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.tanh(py_x)
    torch_y = torch_func.tanh(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: tanh"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: tanh"

def test_expit():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.expit(py_x)
    torch_y = torch_func.expit(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: expit"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: expit"

def test_reciprocal():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.reciprocal(py_x)
    torch_y = torch_func.reciprocal(torch_x)

    assert_close(py_y, torch_y, "reciprocal")

def test_atanh():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = 2 * py_rand.rand(shape) - 1
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.atanh(py_x)
    torch_y = torch_func.atanh(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    # the atanh function is a bit less precise than the others
    # so the tolerance is a bit more flexible
    assert py_matrix.allclose(py_y, py_torch_y, rtol=1e-05, atol=1e-07), \
    "python -> torch -> python failure: atanh"

    assert torch_matrix.allclose(torch_y, torch_py_y, rtol=1e-05, atol=1e-07), \
    "torch -> python -> torch failure: atanh"

def test_sqrt():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.sqrt(py_x)
    torch_y = torch_func.sqrt(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: sqrt"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: sqrt"

def test_square():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.square(py_x)
    torch_y = torch_func.square(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: square"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: square"

def test_tpow():
    shape = (100, 100)
    power = 3
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.tpow(py_x, power)
    torch_y = torch_func.tpow(torch_x, power)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: tpow"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: tpow"

def test_cosh():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.cosh(py_x)
    torch_y = torch_func.cosh(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: cosh"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: cosh"

def test_logaddexp():
    shape = (100, 100)
    py_rand.set_seed()
    py_x_1 = py_rand.randn(shape)
    py_x_2 = py_rand.randn(shape)

    torch_x_1 = torch_matrix.float_tensor(py_x_1)
    torch_x_2 = torch_matrix.float_tensor(py_x_2)

    py_y = py_func.logaddexp(py_x_1, py_x_2)
    torch_y = torch_func.logaddexp(torch_x_1, torch_x_2)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: cosh"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: cosh"

def test_logcosh():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.logcosh(py_x)
    torch_y = torch_func.logcosh(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: logcosh"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: logcosh"

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

def test_logit():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.logit(py_x)
    torch_y = torch_func.logit(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: logit"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: logit"

def test_softplus():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.softplus(py_x)
    torch_y = torch_func.softplus(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: softplus"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: softplus"

def test_cos():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.cos(py_x)
    torch_y = torch_func.cos(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: cos"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: cos"

def test_sin():
    shape = (100, 100)
    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.sin(py_x)
    torch_y = torch_func.sin(torch_x)

    torch_py_y = torch_matrix.float_tensor(py_y)
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "python -> torch -> python failure: sin"

    assert torch_matrix.allclose(torch_y, torch_py_y), \
    "torch -> python -> torch failure: sin"


if __name__ == "__main__":
    pytest.main([__file__])
