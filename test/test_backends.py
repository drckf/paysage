import paysage.backends.python_backend.matrix as py_matrix
import paysage.backends.python_backend.nonlinearity as py_func
import paysage.backends.python_backend.rand as py_rand
import paysage.backends.python_backend.typedef as py_typedef

import paysage.backends.pytorch_backend.matrix as torch_matrix
import paysage.backends.pytorch_backend.nonlinearity as torch_func
import paysage.backends.pytorch_backend.rand as torch_rand
import paysage.backends.pytorch_backend.typedef as torch_typedef

import paysage.backends as be
import paysage.math_utils as mu

from numpy import allclose
from scipy import special
import numpy as np
import pytest

# ---- testing utility functions ----- #

def assert_close(pymat, torchmat, name, rtol=1e-05, atol=1e-06):

    pytorchmat = torch_matrix.to_numpy_array(torchmat)
    torchpymat = torch_matrix.float_tensor(pymat)

    py_vs_torch = py_matrix.allclose(pymat, pytorchmat, rtol=rtol, atol=atol)
    torch_vs_py = torch_matrix.allclose(torchmat, torchpymat, rtol=rtol, atol=atol)

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

def compare_lists(a, b):
    return all([be.allclose(ai, bi) for ai, bi in zip(a, b)])


# ----- Tests ------ #


def test_construction():

    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)
    py_torch_x = torch_matrix.to_numpy_array(torch_x)

    assert py_matrix.allclose(py_x, py_torch_x), \
    "torch float constructor failure"

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    py_y = py_matrix.float_tensor(py_x)

    assert py_matrix.allclose(py_x, py_y), \
    "python float constructor failure"

    py_rand.set_seed()
    py_x = py_rand.rand_int(0,10,shape)
    torch_x = torch_matrix.long_tensor(py_x)
    py_torch_x = torch_matrix.to_numpy_array(torch_x)

    assert py_matrix.allclose(py_x, py_torch_x), \
    "torch long constructor failure"

    py_rand.set_seed()
    py_x = py_rand.rand_int(0,10,shape)
    py_y = py_matrix.long_tensor(py_x)

    assert py_matrix.allclose(py_x, py_y), \
    "python long constructor failure"

    py_rand.set_seed()
    py_x = py_rand.rand((10,)).astype(float)
    py_x = [p for p in py_x]
    torch_x = torch_matrix.float_tensor(py_x)
    py_y = py_matrix.float_tensor(py_x)
    py_torch_x = torch_matrix.to_numpy_array(torch_x)

    assert py_matrix.allclose(py_y, py_torch_x), \
    "constructor from float list failure"

    py_rand.set_seed()
    py_x = py_rand.rand_int(0, 10, (10,))
    py_x = [int(p) for p in py_x]
    torch_x = torch_matrix.long_tensor(py_x)
    py_y = py_matrix.long_tensor(py_x)
    py_torch_x = torch_matrix.to_numpy_array(torch_x)

    assert py_matrix.allclose(py_y, py_torch_x), \
    "constructor from int list failure"


def test_conversion():

    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.rand_int(0,10,shape)
    py_y = py_matrix.cast_float(py_x)
    torch_y = torch_matrix.cast_float(torch_matrix.long_tensor(py_x))
    py_torch_y = torch_matrix.to_numpy_array(torch_y)

    assert py_matrix.allclose(py_y, py_torch_y), \
    "long to float conversion failure"

def test_indexing():

    shape = (100,)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    index = [0, 2, 10]
    py_index = py_matrix.long_tensor(index)
    torch_index = torch_matrix.long_tensor(index)

    py_subset = py_x[py_index]
    torch_subset = torch_x[torch_index]

    assert_close(py_subset, torch_subset, "indexing")

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

def test_unsqueeze():

    shape = (100,)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_x_col = py_matrix.unsqueeze(py_x,1)
    torch_x_col = torch_matrix.unsqueeze(torch_x,1)
    py_x_row = py_matrix.unsqueeze(py_x,0)
    torch_x_row = torch_matrix.unsqueeze(torch_x,0)

    assert_close(py_x_col, torch_x_col, \
    "python->torch failure: unsqueeze to col vector")

    assert_close(py_x_row, torch_x_row, \
    "python->torch failure: unsqueeze to row vector")

    assert py_matrix.shape(py_x_col) == py_matrix.shape(py_x_row.transpose())
    assert torch_matrix.shape(torch_x_col) == torch_matrix.shape(torch_matrix.transpose(torch_x_row))

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
    py_matrix.fill_diagonal_(py_mat, fill_value)

    assert py_matrix.allclose(py_mat, py_mult), \
    "python fill != python multiplly for diagonal matrix"

    torch_mult = fill_value * torch_mat
    torch_matrix.fill_diagonal_(torch_mat, fill_value)

    assert torch_matrix.allclose(torch_mat, torch_mult), \
    "torch fill != python multiplly for diagonal matrix"

    assert_close(py_mat, torch_mat, "fill_diagonal_")

def test_scatter():

    n = 10

    py_mat = py_matrix.zeros((n, n))
    torch_mat = torch_matrix.zeros((n, n))

    value = 2.0
    py_inds = [0, 7, 5, 3, 2, 1, 5, 8, 4, 2]
    torch_inds = torch_matrix.long_tensor(py_inds)

    py_matrix.scatter_(py_mat, py_inds, value)
    torch_matrix.scatter_(torch_mat, torch_inds, value)

    assert_close(py_mat, torch_mat, "scatter_")

def test_index_select():
    shape = (100, 100)

    py_rand.set_seed()

    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_inds = py_matrix.long_tensor([0, 7, 5, 3, 2, 1, 5, 8, 4, 2])
    torch_inds = torch_matrix.long_tensor(py_inds)

    # dimension 0
    py_select = py_matrix.index_select(py_mat, py_inds, 0)
    torch_select = torch_matrix.index_select(torch_mat, torch_inds, 0)
    assert_close(py_select, torch_select, "index_select: dim = 0")

    # dimension 1
    py_select = py_matrix.index_select(py_mat, py_inds, 1)
    torch_select = torch_matrix.index_select(torch_mat, torch_inds, 1)
    assert_close(py_select, torch_select, "index_select: dim = 1")

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

def test_clip_():

    shape = (100,100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # test two sided clip
    py_matrix.clip_(py_mat, a_min=0, a_max=1)
    torch_matrix.clip_(torch_mat, a_min=0, a_max=1)

    assert_close(py_mat, torch_mat, "clip_ (two-sided)")

    # test lower clip
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_matrix.clip_(py_mat, a_min=0)
    torch_matrix.clip_(torch_mat, a_min=0)

    assert_close(py_mat, torch_mat, "clip_ (lower)")

    # test upper clip
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_matrix.clip_(py_mat, a_max=1)
    torch_matrix.clip_(torch_mat, a_max=1)

    assert_close(py_mat, torch_mat, "clip_ (upper)")

def test_tclip():

    shape = (100,100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_min = py_rand.rand(shape) - 1.0
    torch_min = torch_matrix.float_tensor(py_min)

    py_max = py_rand.rand(shape) + 1.0
    torch_max = torch_matrix.float_tensor(py_max)

    # test two sided clip
    py_clipped = py_matrix.tclip(py_mat, py_min, py_max)
    torch_clipped = torch_matrix.tclip(torch_mat, torch_min, torch_max)
    assert_close(py_clipped, torch_clipped, "tclip (two-sided)")

    # test lower clip
    py_clipped = py_matrix.tclip(py_mat, py_min)
    torch_clipped = torch_matrix.tclip(torch_mat, torch_min)
    assert_close(py_clipped, torch_clipped, "tclip (lower)")

    # test upper clip
    py_clipped = py_matrix.tclip(py_mat, a_max=py_max)
    torch_clipped = torch_matrix.tclip(torch_mat, a_max=torch_max)
    assert_close(py_clipped, torch_clipped, "tclip (upper)")

def test_tclip_():

    shape = (100,100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_min = py_rand.rand(shape) - 1.0
    torch_min = torch_matrix.float_tensor(py_min)

    py_max = py_rand.rand(shape) + 1.0
    torch_max = torch_matrix.float_tensor(py_max)

    # test two sided clip
    py_matrix.tclip_(py_mat, py_min, py_max)
    torch_matrix.tclip_(torch_mat, torch_min, torch_max)

    assert_close(py_mat, torch_mat, "tclip_ (two-sided)")

    # test lower clip
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_matrix.tclip_(py_mat, py_min)
    torch_matrix.tclip_(torch_mat, torch_min)

    assert_close(py_mat, torch_mat, "tclip_ (lower)")

    # test upper clip
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_matrix.tclip_(py_mat, a_max=py_max)
    torch_matrix.tclip_(torch_mat, a_max=torch_max)

    assert_close(py_mat, torch_mat, "tclip_ (upper)")

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

def test_mix_():
    shape = (100,100)

    # single mixing coefficient
    torch_w = 0.1
    py_w = py_matrix.float_scalar(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.mix_(py_w, py_x, py_y)
    torch_matrix.mix_(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "mix_: float w")

    # matrix of mixing coefficients
    torch_w = torch_rand.rand(shape)
    py_w = torch_matrix.to_numpy_array(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.mix_(py_w, py_x, py_y)
    torch_matrix.mix_(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "mix_: matrix w")

def test_mix():
    shape = (100,100)

    # single mixing coefficient
    torch_w = 0.1
    py_w = py_matrix.float_scalar(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.mix_(py_w, py_x, py_y)
    torch_matrix.mix_(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "mix: float w")

    # matrix of mixing coefficients
    torch_w = torch_rand.rand(shape)
    py_w = torch_matrix.to_numpy_array(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.mix_(py_w, py_x, py_y)
    torch_matrix.mix_(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "mix: matrix w")


def test_square_mix_():
    shape = (100,100)

    # single mixing coefficient
    torch_w = 0.1
    py_w = py_matrix.float_scalar(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.square_mix_(py_w, py_x, py_y)
    torch_matrix.square_mix_(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "square_mix_: float w")

    # single mixing coefficient
    torch_w = torch_rand.rand(shape)
    py_w = torch_matrix.to_numpy_array(torch_w)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_matrix.square_mix_(py_w, py_x, py_y)
    torch_matrix.square_mix_(torch_w, torch_x, torch_y)

    assert_close(py_x, torch_x, "square_mix_: matrix w")

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
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    # overall norm
    py_norm = py_matrix.norm(py_x)
    torch_norm = torch_matrix.norm(torch_x)

    assert allclose(py_norm, torch_norm), \
    "python l2 norm != torch l2 norm"

    # norm over axis 0, keepdims = False
    py_norm = py_matrix.norm(py_x, axis=0)
    torch_norm = torch_matrix.norm(torch_x, axis=0)
    assert_close(py_norm, torch_norm, "norm (axis-0)")

    # norm over axis 0, keepdims
    py_norm = py_matrix.norm(py_x, axis=0, keepdims=True)
    torch_norm = torch_matrix.norm(torch_x, axis=0, keepdims=True)
    assert_close(py_norm, torch_norm, "norm (axis-0, keepdims)")

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
    assert_close(py_max, torch_max, "tmax (axis-0)")

    # max over axis 1
    py_max = py_matrix.tmax(py_mat, axis=1)
    torch_max = torch_matrix.tmax(torch_mat, axis=1)
    assert_close(py_max, torch_max, "tmax (axis-1)")

    # max over axis 0, keepdims = True
    py_max = py_matrix.tmax(py_mat, axis=0, keepdims=True)
    torch_max = torch_matrix.tmax(torch_mat, axis=0, keepdims=True)
    assert_close(py_max, torch_max, "tmax (axis-0, keepdims)")

    # max over axis 1, keepdims = True
    py_max = py_matrix.tmax(py_mat, axis=1, keepdims=True)
    torch_max = torch_matrix.tmax(torch_mat, axis=1, keepdims=True)
    assert_close(py_max, torch_max, "tmax (axis-1, keepdims)")

def test_tmin():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # overall min
    py_min = py_matrix.tmin(py_mat)
    torch_min = torch_matrix.tmin(torch_mat)

    assert allclose(py_min, torch_min), \
    "python overal min != torch overall min"

    # min over axis 0
    py_min = py_matrix.tmin(py_mat, axis=0)
    torch_min = torch_matrix.tmin(torch_mat, axis=0)
    assert_close(py_min, torch_min, "tmin (axis-0)")

    # min over axis 1
    py_min = py_matrix.tmin(py_mat, axis=1)
    torch_min = torch_matrix.tmin(torch_mat, axis=1)
    assert_close(py_min, torch_min, "tmin (axis-1)")

    # min over axis 0, keepdims = True
    py_min = py_matrix.tmin(py_mat, axis=0, keepdims=True)
    torch_min = torch_matrix.tmin(torch_mat, axis=0, keepdims=True)
    assert_close(py_min, torch_min, "tmin (axis-0, keepdims)")

    # min over axis 1, keepdims = True
    py_min = py_matrix.tmin(py_mat, axis=1, keepdims=True)
    torch_min = torch_matrix.tmin(torch_mat, axis=1, keepdims=True)
    assert_close(py_min, torch_min, "tmin (axis-1, keepdims)")

def test_mean():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # overall mean
    py_mean = py_matrix.mean(py_mat)
    torch_mean = torch_matrix.mean(torch_mat)

    assert allclose(py_mean, torch_mean), \
    "python overal mean != torch overall mean"

    # mean over axis 0
    py_mean = py_matrix.mean(py_mat, axis=0)
    torch_mean = torch_matrix.mean(torch_mat, axis=0)
    assert_close(py_mean, torch_mean, "mean (axis-0)")

    # mean over axis 1
    py_mean = py_matrix.mean(py_mat, axis=1)
    torch_mean = torch_matrix.mean(torch_mat, axis=1)
    assert_close(py_mean, torch_mean, "mean (axis-1)")

    # mean over axis 0, keepdims = True
    py_mean = py_matrix.mean(py_mat, axis=0, keepdims=True)
    torch_mean = torch_matrix.mean(torch_mat, axis=0, keepdims=True)
    assert_close(py_mean, torch_mean, "mean (axis-0, keepdims)")

    # mean over axis 1, keepdims = True
    py_mean = py_matrix.mean(py_mat, axis=1, keepdims=True)
    torch_mean = torch_matrix.mean(torch_mat, axis=1, keepdims=True)
    assert_close(py_mean, torch_mean, "mean (axis-1, keepdims)")

def test_center():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # center over axis 0
    py_centered = py_matrix.center(py_mat, axis=0)
    torch_centered = torch_matrix.center(torch_mat, axis=0)
    assert_close(py_centered, torch_centered, "center (axis-0)")

    # center over axis 1
    py_centered = py_matrix.center(py_mat, axis=1)
    torch_centered = torch_matrix.center(torch_mat, axis=1)
    assert_close(py_centered, torch_centered, "center (axis-1)")

def test_var():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # overall var
    py_var = py_matrix.var(py_mat)
    torch_var = torch_matrix.var(torch_mat)

    assert allclose(py_var, torch_var), \
    "python overal var != torch overall var"

    # var over axis 0
    py_var = py_matrix.var(py_mat, axis=0)
    torch_var = torch_matrix.var(torch_mat, axis=0)
    assert_close(py_var, torch_var, "var (axis-0)")

    # var over axis 1
    py_var = py_matrix.var(py_mat, axis=1)
    torch_var = torch_matrix.var(torch_mat, axis=1)
    assert_close(py_var, torch_var, "var (axis-1)")

    # var over axis 0, keepdims = True
    py_var = py_matrix.var(py_mat, axis=0, keepdims=True)
    torch_var = torch_matrix.var(torch_mat, axis=0, keepdims=True)
    assert_close(py_var, torch_var, "var (axis-0, keepdims)")

    # var over axis 1, keepdims = True
    py_var = py_matrix.var(py_mat, axis=1, keepdims=True)
    torch_var = torch_matrix.var(torch_mat, axis=1, keepdims=True)
    assert_close(py_var, torch_var, "var (axis-1, keepdims)")

def test_std():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # overall std
    py_std = py_matrix.std(py_mat)
    torch_std = torch_matrix.std(torch_mat)

    assert allclose(py_std, torch_std), \
    "python overal std != torch overall std"

    # std over axis 0
    py_std = py_matrix.std(py_mat, axis=0)
    torch_std = torch_matrix.std(torch_mat, axis=0)
    assert_close(py_std, torch_std, "std (axis-0)")

    # std over axis 1
    py_std = py_matrix.std(py_mat, axis=1)
    torch_std = torch_matrix.std(torch_mat, axis=1)
    assert_close(py_std, torch_std, "std (axis-1)")

    # std over axis 0, keepdims = True
    py_std = py_matrix.std(py_mat, axis=0, keepdims=True)
    torch_std = torch_matrix.std(torch_mat, axis=0, keepdims=True)
    assert_close(py_std, torch_std, "std (axis-0, keepdims)")

    # std over axis 1, keepdims = True
    py_std = py_matrix.std(py_mat, axis=1, keepdims=True)
    torch_std = torch_matrix.std(torch_mat, axis=1, keepdims=True)
    assert_close(py_std, torch_std, "std (axis-1, keepdims)")

def test_cov():
    shape_x = (100, 10)
    shape_y = (100, 10)

    py_rand.set_seed()

    py_x = py_rand.randn(shape_x)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_rand.randn(shape_y)
    torch_y = torch_matrix.float_tensor(py_y)

    py_cov = py_matrix.cov(py_x, py_y)
    torch_cov = torch_matrix.cov(torch_x, torch_y)
    assert_close(py_cov, torch_cov, "cov")

def test_corr():
    shape_x = (100, 10)
    shape_y = (100, 10)

    py_rand.set_seed()

    py_x = py_rand.randn(shape_x)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_rand.randn(shape_y)
    torch_y = torch_matrix.float_tensor(py_y)

    py_corr = py_matrix.corr(py_x, py_y)
    torch_corr = torch_matrix.corr(torch_x, torch_y)
    assert_close(py_corr, torch_corr, "corr")

def test_tsum():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # overall tsum
    py_tsum = py_matrix.tsum(py_mat)
    torch_tsum = torch_matrix.tsum(torch_mat)

    assert allclose(py_tsum, torch_tsum), \
    "python overal tsum != torch overall tsum"

    # tsum over axis 0
    py_tsum = py_matrix.tsum(py_mat, axis=0)
    torch_tsum = torch_matrix.tsum(torch_mat, axis=0)
    assert_close(py_tsum, torch_tsum, "tsum (axis-0)")

    # tsum over axis 1
    py_tsum = py_matrix.tsum(py_mat, axis=1)
    torch_tsum = torch_matrix.tsum(torch_mat, axis=1)
    assert_close(py_tsum, torch_tsum, "tsum (axis-1)")

    # tsum over axis 0, keepdims = True
    py_tsum = py_matrix.tsum(py_mat, axis=0, keepdims=True)
    torch_tsum = torch_matrix.tsum(torch_mat, axis=0, keepdims=True)
    assert_close(py_tsum, torch_tsum, "tsum (axis-0, keepdims)")

    # tsum over axis 1, keepdims = True
    py_tsum = py_matrix.tsum(py_mat, axis=1, keepdims=True)
    torch_tsum = torch_matrix.tsum(torch_mat, axis=1, keepdims=True)
    assert_close(py_tsum, torch_tsum, "tsum (axis-1, keepdims)")

def test_tprod():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # overall tprod
    py_tprod = py_matrix.tprod(py_mat)
    torch_tprod = torch_matrix.tprod(torch_mat)

    assert allclose(py_tprod, torch_tprod), \
    "python overal tprod != torch overall tprod"

    # tprod over axis 0
    py_tprod = py_matrix.tprod(py_mat, axis=0)
    torch_tprod = torch_matrix.tprod(torch_mat, axis=0)
    assert_close(py_tprod, torch_tprod, "tprod (axis-0)")

    # tprod over axis 1
    py_tprod = py_matrix.tprod(py_mat, axis=1)
    torch_tprod = torch_matrix.tprod(torch_mat, axis=1)
    assert_close(py_tprod, torch_tprod, "tprod (axis-1)")

    # tprod over axis 0, keepdims = True
    py_tprod = py_matrix.tprod(py_mat, axis=0, keepdims=True)
    torch_tprod = torch_matrix.tprod(torch_mat, axis=0, keepdims=True)
    assert_close(py_tprod, torch_tprod, "tprod (axis-0, keepdims)")

    # tprod over axis 1, keepdims = True
    py_tprod = py_matrix.tprod(py_mat, axis=1, keepdims=True)
    torch_tprod = torch_matrix.tprod(torch_mat, axis=1, keepdims=True)
    assert_close(py_tprod, torch_tprod, "tprod (axis-1, keepdims)")

def test_equal():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_eq = py_matrix.equal(py_x, py_y)
    torch_eq = torch_matrix.equal(torch_x, torch_y)
    py_torch_eq = torch_matrix.to_numpy_array(torch_eq)

    assert py_matrix.allclose(py_eq, py_torch_eq), \
    "python equal != torch equal"

def test_not_equal():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_neq = py_matrix.not_equal(py_x, py_y)
    torch_neq = torch_matrix.not_equal(torch_x, torch_y)
    py_torch_neq = torch_matrix.to_numpy_array(torch_neq)

    assert py_matrix.allclose(py_neq, py_torch_neq), \
    "python not equal != torch not equal"

def test_greater():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.greater(py_x, py_y)
    torch_res = torch_matrix.greater(torch_x, torch_y)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python greater != torch greater"

def test_greater_equal():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.greater_equal(py_x, py_y)
    torch_res = torch_matrix.greater_equal(torch_x, torch_y)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python greater_equal != torch greater_equal"

def test_lesser():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.lesser(py_x, py_y)
    torch_res = torch_matrix.lesser(torch_x, torch_y)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python lesser != torch lesser"

def test_lesser_equal():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.lesser_equal(py_x, py_y)
    torch_res = torch_matrix.lesser_equal(torch_x, torch_y)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python lesser_equal != torch lesser_equal"

def test_tany():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.lesser_equal(py_x, py_y)
    torch_res = torch_matrix.lesser_equal(torch_x, torch_y)

    # overall
    py_any = py_matrix.tany(py_res)
    torch_any = torch_matrix.tany(torch_res)
    assert py_any == torch_any, \
    "python tany != torch tany: overall"

    # axis = 0
    py_any = py_matrix.tany(py_res, axis=0)
    torch_any = torch_matrix.tany(torch_res, axis=0)
    py_torch_any = torch_matrix.to_numpy_array(torch_any)

    assert py_matrix.allclose(py_any, py_torch_any), \
    "python tany != torch tany: (axis-0)"

    # axis = 1
    py_any = py_matrix.tany(py_res, axis=1)
    torch_any = torch_matrix.tany(torch_res, axis=1)
    py_torch_any = torch_matrix.to_numpy_array(torch_any)

    assert py_matrix.allclose(py_any, py_torch_any), \
    "python tany != torch tany: (axis-1)"

    # axis = 0, keepdims
    py_any = py_matrix.tany(py_res, axis=0, keepdims=True)
    torch_any = torch_matrix.tany(torch_res, axis=0, keepdims=True)
    py_torch_any = torch_matrix.to_numpy_array(torch_any)

    assert py_matrix.allclose(py_any, py_torch_any), \
    "python tany != torch tany: (axis-0, keepdims)"

    # axis = 1, keepdims
    py_any = py_matrix.tany(py_res, axis=1, keepdims=True)
    torch_any = torch_matrix.tany(torch_res, axis=1, keepdims=True)
    py_torch_any = torch_matrix.to_numpy_array(torch_any)

    assert py_matrix.allclose(py_any, py_torch_any), \
    "python tany != torch tany: (axis-1, keepdim)"

def test_tall():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.lesser_equal(py_x, py_y)
    torch_res = torch_matrix.lesser_equal(torch_x, torch_y)

    # overall
    py_all = py_matrix.tall(py_res)
    torch_all = torch_matrix.tall(torch_res)
    assert py_all == torch_all, \
    "python tall != torch tall: overall"

    # axis = 0
    py_all = py_matrix.tall(py_res, axis=0)
    torch_all = torch_matrix.tall(torch_res, axis=0)
    py_torch_all = torch_matrix.to_numpy_array(torch_all)

    assert py_matrix.allclose(py_all, py_torch_all), \
    "python tall != torch tall: (axis-0)"

    # axis = 1
    py_all = py_matrix.tall(py_res, axis=1)
    torch_all = torch_matrix.tall(torch_res, axis=1)
    py_torch_all = torch_matrix.to_numpy_array(torch_all)

    assert py_matrix.allclose(py_all, py_torch_all), \
    "python tall != torch tall: (axis-1)"

    # axis = 0, keepdims
    py_all = py_matrix.tall(py_res, axis=0, keepdims=True)
    torch_all = torch_matrix.tall(torch_res, axis=0, keepdims=True)
    py_torch_all = torch_matrix.to_numpy_array(torch_all)

    assert py_matrix.allclose(py_all, py_torch_all), \
    "python tall != torch tall: (axis-0, keepdims)"

    # axis = 1, keepdims
    py_all = py_matrix.tall(py_res, axis=1, keepdims=True)
    torch_all = torch_matrix.tall(torch_res, axis=1, keepdims=True)
    py_torch_all = torch_matrix.to_numpy_array(torch_all)

    assert py_matrix.allclose(py_all, py_torch_all), \
    "python tall != torch tall: (axis-1, keepdim)"

def test_maximum():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.maximum(py_x, py_y)
    torch_res = torch_matrix.maximum(torch_x, torch_y)

    assert_close(py_res, torch_res, "maximum")

def test_minimum():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    py_y = py_rand.randn(shape)

    torch_x = torch_matrix.float_tensor(py_x)
    torch_y = torch_matrix.float_tensor(py_y)

    py_res = py_matrix.minimum(py_x, py_y)
    torch_res = torch_matrix.minimum(torch_x, torch_y)

    assert_close(py_res, torch_res, "minimum")

def test_argsort():
    shape = (100,)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_res = py_matrix.argsort(py_x)
    torch_res = torch_matrix.argsort(torch_x)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python argsort != torch argsort"

def test_sort():
    shape = (100,)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_res = py_matrix.sort(py_x)
    torch_res = torch_matrix.sort(torch_x)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python sort != torch sort"

def test_argmax():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    # axis=0
    py_res = py_matrix.argmax(py_x, axis=0)
    torch_res = torch_matrix.argmax(torch_x, axis=0)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python argmax != torch argmax: (axis-0)"

    # axis=1
    py_res = py_matrix.argmax(py_x, axis=1)
    torch_res = torch_matrix.argmax(torch_x, axis=1)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python argmax != torch argmax: (axis-1)"

def test_argmin():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    # axis=0
    py_res = py_matrix.argmin(py_x, axis=0)
    torch_res = torch_matrix.argmin(torch_x, axis=0)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python argmin != torch argmin: (axis-0)"

    # axis=1
    py_res = py_matrix.argmin(py_x, axis=1)
    torch_res = torch_matrix.argmin(torch_x, axis=1)
    py_torch_res = torch_matrix.to_numpy_array(torch_res)

    assert py_matrix.allclose(py_res, py_torch_res), \
    "python argmin != torch argmin: (axis-1)"

def test_dot():
    # vector-vector
    a_shape = (100,)
    b_shape = (100,)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_dot = py_matrix.dot(py_a, py_b)
    torch_dot = torch_matrix.dot(torch_a, torch_b)

    assert allclose(py_dot, torch_dot, 1e-4, 1e-4), \
    "python dot != torch_dot: vector-vector"

    # matrix-vector
    a_shape = (100,100)
    b_shape = (100,)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_dot = py_matrix.dot(py_a, py_b)
    torch_dot = torch_matrix.dot(torch_a, torch_b)

    # occasionally fails without a looser threshold
    assert_close(py_dot, torch_dot, "dot: matrix-vector", 1e-4, 1e-4)

    # matrix-matrix
    a_shape = (100,100)
    b_shape = (100,100)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_dot = py_matrix.dot(py_a, py_b)
    torch_dot = torch_matrix.dot(torch_a, torch_b)

    # occasionally fails without a looser threshold
    assert_close(py_dot, torch_dot, "dot: matrix-matrix", 1e-4, 1e-4)

def test_outer():
    a_shape = (100,)
    b_shape = (100,)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_res = py_matrix.outer(py_a, py_b)
    torch_res = torch_matrix.outer(torch_a, torch_b)

    assert_close(py_res, torch_res, "outer")

def test_python_broadcast():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)

    numpy_product = py_a * py_b
    broadcast_product = py_matrix.broadcast(py_a, py_b) * py_b

    assert allclose(numpy_product, broadcast_product), \
    "python broadcast failure: (N, 1) x (N, M)"

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)

    numpy_product = py_a * py_b
    broadcast_product = py_matrix.broadcast(py_a, py_b) * py_b

    assert allclose(numpy_product, broadcast_product), \
    "python broadcast failure: (1, N) x (M, N)"

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)

    numpy_product = py_a * py_b
    broadcast_product = py_matrix.broadcast(py_a, py_b) * py_b

    assert allclose(numpy_product, broadcast_product), \
    "python broadcast failure: (N,) x (M, N)"

def test_broadcast():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_broadcast = py_matrix.broadcast(py_a, py_b)
    torch_broadcast = torch_matrix.broadcast(torch_a, torch_b)
    assert_close(py_broadcast, torch_broadcast, "broadcast: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_broadcast = py_matrix.broadcast(py_a, py_b)
    torch_broadcast = torch_matrix.broadcast(torch_a, torch_b)
    assert_close(py_broadcast, torch_broadcast, "broadcast: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_broadcast = py_matrix.broadcast(py_a, py_b)
    torch_broadcast = torch_matrix.broadcast(torch_a, torch_b)
    assert_close(py_broadcast, torch_broadcast, "broadcast: (N,) x (M, N)")

def test_add():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_add = py_matrix.add(py_a, py_b)
    torch_add = torch_matrix.add(torch_a, torch_b)
    assert_close(py_add, torch_add, "add: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_add = py_matrix.add(py_a, py_b)
    torch_add = torch_matrix.add(torch_a, torch_b)
    assert_close(py_add, torch_add, "add: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_add = py_matrix.add(py_a, py_b)
    torch_add = torch_matrix.add(torch_a, torch_b)
    assert_close(py_add, torch_add, "add: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_add = py_matrix.add(py_a, py_b)
    torch_add = torch_matrix.add(torch_a, torch_b)
    assert_close(py_add, torch_add, "add: (M, N) x (M, N)")

def test_add_():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.add_(py_a, py_b)
    torch_matrix.add_(torch_a, torch_b)
    assert_close(py_b, torch_b, "add_: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.add_(py_a, py_b)
    torch_matrix.add_(torch_a, torch_b)
    assert_close(py_b, torch_b, "add_: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.add_(py_a, py_b)
    torch_matrix.add_(torch_a, torch_b)
    assert_close(py_b, torch_b, "add_: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.add_(py_a, py_b)
    torch_matrix.add_(torch_a, torch_b)
    assert_close(py_b, torch_b, "add_: (M, N) x (M, N)")

def test_subtract():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_subtract = py_matrix.subtract(py_a, py_b)
    torch_subtract = torch_matrix.subtract(torch_a, torch_b)
    assert_close(py_subtract, torch_subtract, "subtract: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_subtact = py_matrix.subtract(py_a, py_b)
    torch_subtract = torch_matrix.subtract(torch_a, torch_b)
    assert_close(py_subtact, torch_subtract, "subtract: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_subtract = py_matrix.subtract(py_a, py_b)
    torch_subtract = torch_matrix.subtract(torch_a, torch_b)
    assert_close(py_subtract, torch_subtract, "subtract: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_subtract = py_matrix.subtract(py_a, py_b)
    torch_subtract = torch_matrix.subtract(torch_a, torch_b)
    assert_close(py_subtract, torch_subtract, "subtract: (M, N) x (M, N)")

def test_subtract_():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.subtract_(py_a, py_b)
    torch_matrix.subtract_(torch_a, torch_b)
    assert_close(py_b, torch_b, "subtract_: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.subtract_(py_a, py_b)
    torch_matrix.subtract_(torch_a, torch_b)
    assert_close(py_b, torch_b, "subtract_: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.subtract_(py_a, py_b)
    torch_matrix.subtract_(torch_a, torch_b)
    assert_close(py_b, torch_b, "subtract_: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.subtract_(py_a, py_b)
    torch_matrix.subtract_(torch_a, torch_b)
    assert_close(py_b, torch_b, "subtract_: (M, N) x (M, N)")

def test_multiply():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_multiply = py_matrix.multiply(py_a, py_b)
    torch_multiply = torch_matrix.multiply(torch_a, torch_b)
    assert_close(py_multiply, torch_multiply, "multiply: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_multiply = py_matrix.multiply(py_a, py_b)
    torch_multiply = torch_matrix.multiply(torch_a, torch_b)
    assert_close(py_multiply, torch_multiply, "multiply: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_multiply = py_matrix.multiply(py_a, py_b)
    torch_multiply = torch_matrix.multiply(torch_a, torch_b)
    assert_close(py_multiply, torch_multiply, "multiply: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_multiply = py_matrix.multiply(py_a, py_b)
    torch_multiply = torch_matrix.multiply(torch_a, torch_b)
    assert_close(py_multiply, torch_multiply, "multiply: (M, N) x (M, N)")

def test_multiply_():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.multiply_(py_a, py_b)
    torch_matrix.multiply_(torch_a, torch_b)
    assert_close(py_b, torch_b, "multiply_: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.multiply_(py_a, py_b)
    torch_matrix.multiply_(torch_a, torch_b)
    assert_close(py_b, torch_b, "multiply_: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.multiply_(py_a, py_b)
    torch_matrix.multiply_(torch_a, torch_b)
    assert_close(py_b, torch_b, "multiply_: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.multiply_(py_a, py_b)
    torch_matrix.multiply_(torch_a, torch_b)
    assert_close(py_b, torch_b, "multiply_: (M, N) x (M, N)")

def test_divide():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_divide = py_matrix.divide(py_a, py_b)
    torch_divide = torch_matrix.divide(torch_a, torch_b)
    assert_close(py_divide, torch_divide, "divide: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_divide = py_matrix.divide(py_a, py_b)
    torch_divide = torch_matrix.divide(torch_a, torch_b)
    assert_close(py_divide, torch_divide, "divide: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_divide = py_matrix.divide(py_a, py_b)
    torch_divide = torch_matrix.divide(torch_a, torch_b)
    assert_close(py_divide, torch_divide, "divide: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_divide = py_matrix.divide(py_a, py_b)
    torch_divide = torch_matrix.divide(torch_a, torch_b)
    assert_close(py_divide, torch_divide, "divide: (M, N) x (M, N)")

def test_divide_():
    N = 100
    M = 50

    # (N, 1) x (N, M)
    a_shape = (N, 1)
    b_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.divide_(py_a, py_b)
    torch_matrix.divide_(torch_a, torch_b)
    assert_close(py_b, torch_b, "divide_: (N, 1) x (N, M)")

    # (1, N) x (M, N)
    a_shape = (1, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.divide_(py_a, py_b)
    torch_matrix.divide_(torch_a, torch_b)
    assert_close(py_b, torch_b, "divide_: (1, N) x (M, N)")

    # (N,) x (M, N)
    a_shape = (N,)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.divide(py_a, py_b)
    torch_matrix.divide(torch_a, torch_b)
    assert_close(py_b, torch_b, "divide_: (N,) x (M, N)")

    # (M, N) x (M, N)
    a_shape = (M, N)
    b_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.rand(a_shape)
    py_b = py_rand.rand(b_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_matrix.divide(py_a, py_b)
    torch_matrix.divide(torch_a, torch_b)
    assert_close(py_b, torch_b, "divide_: (M, N) x (M, N)")

def test_affine():
    # vector-vector-matrix
    N = 100
    M = 50
    a_shape = (N,)
    b_shape = (M,)
    W_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    py_W = py_rand.randn(W_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)
    torch_W = torch_matrix.float_tensor(py_W)

    py_res = py_matrix.affine(py_a, py_b, py_W)
    torch_res = torch_matrix.affine(torch_a, torch_b, torch_W)

    assert_close(py_res, torch_res, "affine: vector, vector, matrix")

    # matrix-matrix-matrix
    batch_size = 10
    N = 100
    M = 50
    a_shape = (batch_size, N)
    b_shape = (batch_size, M)
    W_shape = (M, N)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    py_W = py_rand.randn(W_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)
    torch_W = torch_matrix.float_tensor(py_W)

    py_res = py_matrix.affine(py_a, py_W, py_b)
    torch_res = torch_matrix.affine(torch_a, torch_W, torch_b)

    assert_close(py_res, torch_res, "affine: matrix, matrix, matrix")

def test_quadratic():
    # vector-vector-matrix
    N = 100
    M = 50
    a_shape = (N,)
    b_shape = (M,)
    W_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    py_W = py_rand.randn(W_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)
    torch_W = torch_matrix.float_tensor(py_W)

    py_res = py_matrix.quadratic(py_a, py_b, py_W)
    torch_res = torch_matrix.quadratic(torch_a, torch_b, torch_W)

    assert allclose(py_res, torch_res), \
    "quadratic: vector, vector, matrix: failure"

    # matrix-matrix-matrix
    batch_size = 10
    N = 100
    M = 50
    a_shape = (batch_size, N)
    b_shape = (M, batch_size)
    W_shape = (N, M)

    py_rand.set_seed()
    py_a = py_rand.randn(a_shape)
    py_b = py_rand.randn(b_shape)
    py_W = py_rand.randn(W_shape)

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)
    torch_W = torch_matrix.float_tensor(py_W)

    py_res = py_matrix.quadratic(py_a, py_b, py_W)
    torch_res = torch_matrix.quadratic(torch_a, torch_b, torch_W)

    # needs a lower tolerance to pass than other tests
    assert_close(py_res, torch_res, "quadratic: matrix, matrix, matrix",
                 rtol=1e-4, atol=1e-8)

def test_inv():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.inv(py_mat)
    torch_res = torch_matrix.inv(torch_mat)

    # needs a lower tolerance to pass than other tests
    assert_close(py_res, torch_res, "inv", rtol=1e-4, atol=1e-5)

def test_pinv():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.pinv(py_mat)
    torch_res = torch_matrix.pinv(torch_mat)

    # needs a lower tolerance to pass than other tests
    assert_close(py_res, torch_res, "inv", rtol=1e-3, atol=1e-3)

def test_qr():
    shape = (100, 10)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.qr(py_mat)
    torch_res = torch_matrix.qr(torch_mat)

    # needs a lower tolerance to pass than other tests
    for i in range(2):
        assert_close(py_res[i], torch_res[i], "qr", rtol=1e-4, atol=1e-5)

def test_svd():
    shape = (100, 10)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.svd(py_mat)
    torch_res = torch_matrix.svd(torch_mat)

    # needs a lower tolerance to pass than other tests
    for i in range(3):
        print(py_matrix.shape(py_res[i]), torch_matrix.shape(torch_res[i]))
        assert_close(py_res[i], torch_res[i], "svd", rtol=1e-3, atol=1e-4)

def test_matrix_sqrt():
    shape = (10, 10)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.matrix_sqrt(py_mat)
    torch_res = torch_matrix.matrix_sqrt(torch_mat)

    # needs a lower tolerance to pass than other tests
    assert_close(py_res, torch_res, "matrix_sqrt", rtol=1e-3, atol=1e-4)

def test_logdet():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.logdet(py_mat)
    torch_res = torch_matrix.logdet(torch_mat)

    # needs a lower tolerance to pass than other tests
    assert allclose(py_res, torch_res), "python logdet != torch logdet"

def test_batch_dot():
    L = 10
    N = 100

    py_rand.set_seed()
    py_a = py_rand.randn((L, N))
    py_b = py_rand.randn((L, N))

    torch_a = torch_matrix.float_tensor(py_a)
    torch_b = torch_matrix.float_tensor(py_b)

    py_res = py_matrix.batch_dot(py_a, py_b)
    torch_res = torch_matrix.batch_dot(torch_a, torch_b)

    assert_close(py_res, torch_res, "batch_dot")

def test_batch_quadratic():
    L = 10
    N = 100
    M = 50

    v_shape = (L, N)
    W_shape = (N, M)
    h_shape = (L, M)

    py_rand.set_seed()
    py_v = py_rand.randn(v_shape)
    py_W = py_rand.randn(W_shape)
    py_h = py_rand.randn(h_shape)

    torch_v = torch_matrix.float_tensor(py_v)
    torch_W = torch_matrix.float_tensor(py_W)
    torch_h = torch_matrix.float_tensor(py_h)

    py_res = py_matrix.batch_quadratic(py_v, py_W, py_h)
    torch_res = torch_matrix.batch_quadratic(torch_v, torch_W, torch_h)

    assert_close(py_res, torch_res, "batch_quadratic")

def test_batch_outer():
    L = 10
    N = 100
    M = 50

    v_shape = (L, N)
    h_shape = (L, M)

    py_rand.set_seed()
    py_v = py_rand.randn(v_shape)
    py_h = py_rand.randn(h_shape)

    torch_v = torch_matrix.float_tensor(py_v)
    torch_h = torch_matrix.float_tensor(py_h)

    py_res = py_matrix.batch_outer(py_v, py_h)
    torch_res = torch_matrix.batch_outer(torch_v, torch_h)

    assert_close(py_res, torch_res, "batch_outer")

def test_repeat():
    shape = (100,)
    n_repeats = 5

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_res = py_matrix.repeat(py_x, n_repeats)
    torch_res = torch_matrix.repeat(torch_x, n_repeats)

    assert_close(py_res, torch_res, "repeat")

def test_stack():
    # vector
    shape = (100,)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.stack([py_mat, py_mat], axis=0)
    torch_res = torch_matrix.stack([torch_mat, torch_mat], axis=0)

    assert_close(py_res, torch_res, "stack: vectors, axis=0")

    py_res = py_matrix.stack([py_mat, py_mat], axis=1)
    torch_res = torch_matrix.stack([torch_mat, torch_mat], axis=1)

    assert_close(py_res, torch_res, "stack: vectors, axis=1")

    # matrix
    shape = (100, 100)

    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.stack([py_mat, py_mat], axis=0)
    torch_res = torch_matrix.stack([torch_mat, torch_mat], axis=0)

    assert_close(py_res, torch_res, "stack: matrices, axis=0")

    py_res = py_matrix.stack([py_mat, py_mat], axis=1)
    torch_res = torch_matrix.stack([torch_mat, torch_mat], axis=1)

    assert_close(py_res, torch_res, "stack: matrices, axis=1")

def test_hstack():
    # vector
    shape = (100,)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.hstack([py_mat, py_mat])
    torch_res = torch_matrix.hstack([torch_mat, torch_mat])

    assert_close(py_res, torch_res, "hstack: vectors")

    # matrix
    shape = (100, 100)

    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.hstack([py_mat, py_mat])
    torch_res = torch_matrix.hstack([torch_mat, torch_mat])

    assert_close(py_res, torch_res, "hstack: matrices")

def test_vstack():
    # vector
    shape = (100,)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.vstack([py_mat, py_mat])
    torch_res = torch_matrix.vstack([torch_mat, torch_mat])

    assert_close(py_res, torch_res, "vstack: vectors")

    # matrix
    shape = (100, 100)

    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    py_res = py_matrix.vstack([py_mat, py_mat])
    torch_res = torch_matrix.vstack([torch_mat, torch_mat])

    assert_close(py_res, torch_res, "vstack: matrices")

def test_trange():
    start = 10
    stop = 100
    step = 2

    py_mat = py_matrix.trange(start, stop, step=step)
    torch_mat = torch_matrix.trange(start, stop, step=step)
    py_torch_mat = torch_matrix.to_numpy_array(torch_mat)

    assert allclose(py_mat, py_torch_mat), \
    "trange failure: start=10, stop=100, step=2"

    start = 10
    stop = 100
    step = 7

    py_mat = py_matrix.trange(start, stop, step=step)
    torch_mat = torch_matrix.trange(start, stop, step=step)
    py_torch_mat = torch_matrix.to_numpy_array(torch_mat)

    assert allclose(py_mat, py_torch_mat), \
    "trange failure: start=10, stop=100, step=7"

def test_cumsum():
    shape = (100, 100)

    py_rand.set_seed()
    py_mat = py_rand.randn(shape)
    torch_mat = torch_matrix.float_tensor(py_mat)

    # cumsum over axis 0
    py_cumsum = py_matrix.cumsum(py_mat, axis=0)
    torch_cumsum = torch_matrix.cumsum(torch_mat, axis=0)
    assert_close(py_cumsum, torch_cumsum, "cumsum (axis-0)", rtol=1e-4, atol=1e-5)

    # cumsum over axis 1
    py_cumsum = py_matrix.cumsum(py_mat, axis=1)
    torch_cumsum = torch_matrix.cumsum(torch_mat, axis=1)
    assert_close(py_cumsum, torch_cumsum, "cumsum (axis-1)", rtol=1e-4, atol=1e-5)


def test_logical_not():
    shape = (10, 10)
    py_rand.set_seed()
    x = py_rand.randn(shape)

    pymat = py_matrix.cast_long(x < 0)
    torchmat = torch_matrix.long_tensor(pymat)

    pymat = py_matrix.logical_not(pymat)
    torchmat = torch_matrix.logical_not(torchmat)

    assert_close(py_matrix.cast_float(pymat), torch_matrix.cast_float(torchmat),
                 "logical_not")

def test_logical_and():
    shape = (10, 10)
    py_rand.set_seed()
    x = py_rand.randn(shape)
    y = py_rand.randn(shape)

    pyx = py_matrix.cast_long(x < 0)
    torchx = torch_matrix.long_tensor(pyx)

    pyy = py_matrix.cast_long(y < 0)
    torchy = torch_matrix.long_tensor(pyy)

    pymat = py_matrix.logical_and(pyx, pyy)
    torchmat = torch_matrix.logical_and(torchx, torchy)

    assert_close(py_matrix.cast_float(pymat), torch_matrix.cast_float(torchmat),
                 "logical_and")

def test_logical_or():
    shape = (10, 10)
    py_rand.set_seed()
    x = py_rand.randn(shape)
    y = py_rand.randn(shape)

    pyx = py_matrix.cast_long(x < 0)
    torchx = torch_matrix.long_tensor(pyx)

    pyy = py_matrix.cast_long(y < 0)
    torchy = torch_matrix.long_tensor(pyy)

    pymat = py_matrix.logical_or(pyx, pyy)
    torchmat = torch_matrix.logical_or(torchx, torchy)

    assert_close(py_matrix.cast_float(pymat), torch_matrix.cast_float(torchmat),
                 "logical_and")


# ----- Nonlinearities ----- #

def test_tabs():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.tabs(py_x)
    torch_y = torch_func.tabs(torch_x)
    assert_close(py_y, torch_y, "tabs")

def test_exp():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.exp(py_x)
    torch_y = torch_func.exp(torch_x)
    assert_close(py_y, torch_y, "exp")

def test_log():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.log(py_x)
    torch_y = torch_func.log(torch_x)
    assert_close(py_y, torch_y, "log")

def test_tanh():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.tanh(py_x)
    torch_y = torch_func.tanh(torch_x)
    assert_close(py_y, torch_y, "tanh")

def test_expit():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.expit(py_x)
    torch_y = torch_func.expit(torch_x)
    assert_close(py_y, torch_y, "expit")

def test_softmax():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.softmax(py_x)
    torch_y = torch_func.softmax(torch_x)
    assert_close(py_y, torch_y, "softmax")

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
    # the atanh function is a bit less precise than the others
    # so the tolerance is a bit more flexible
    assert_close(py_y, torch_y, "atanh", rtol=1e-05, atol=1e-07)

def test_sqrt():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.sqrt(py_x)
    torch_y = torch_func.sqrt(torch_x)
    assert_close(py_y, torch_y, "sqrt")

def test_square():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.square(py_x)
    torch_y = torch_func.square(torch_x)
    assert_close(py_y, torch_y, "square")

def test_tpow():
    shape = (100, 100)
    power = 3

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.tpow(py_x, power)
    torch_y = torch_func.tpow(torch_x, power)
    assert_close(py_y, torch_y, "tpow")

def test_cosh():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.cosh(py_x)
    torch_y = torch_func.cosh(torch_x)
    assert_close(py_y, torch_y, "cosh")

def test_logaddexp():
    shape = (100, 100)

    py_rand.set_seed()
    py_x_1 = py_rand.randn(shape)
    py_x_2 = py_rand.randn(shape)

    torch_x_1 = torch_matrix.float_tensor(py_x_1)
    torch_x_2 = torch_matrix.float_tensor(py_x_2)

    py_y = py_func.logaddexp(py_x_1, py_x_2)
    torch_y = torch_func.logaddexp(torch_x_1, torch_x_2)
    assert_close(py_y, torch_y, "logaddexp")

def test_logcosh():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.logcosh(py_x)
    torch_y = torch_func.logcosh(torch_x)
    assert_close(py_y, torch_y, "logcosh")

def test_acosh():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = 1 + py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.acosh(py_x)
    torch_y = torch_func.acosh(torch_x)
    assert_close(py_y, torch_y, "acosh")

def test_logit():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.rand(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.logit(py_x)
    torch_y = torch_func.logit(torch_x)
    assert_close(py_y, torch_y, "logit")

def test_softplus():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.softplus(py_x)
    torch_y = torch_func.softplus(torch_x)
    assert_close(py_y, torch_y, "softplus")

def test_cos():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.cos(py_x)
    torch_y = torch_func.cos(torch_x)
    assert_close(py_y, torch_y, "cos")

def test_sin():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.sin(py_x)
    torch_y = torch_func.sin(torch_x)
    assert_close(py_y, torch_y, "sin")

def test_normal_pdf():
    shape = (100, 100)

    py_rand.set_seed()
    py_x = py_rand.randn(shape)
    torch_x = torch_matrix.float_tensor(py_x)

    py_y = py_func.normal_pdf(py_x)
    torch_y = torch_func.normal_pdf(torch_x)

    assert_close(py_y, torch_y, "normal_pdf")



# ----- Random Sampling ----- #

def test_rand_softmax():
    num_samples = 1000000
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    scale_factor = 3.7 # factor to scale past unitarity
    expected_counts = probs * num_samples

    py_rand.set_seed()

    py_phi_1D = py_func.log(scale_factor * probs)
    py_phi = py_matrix.broadcast(py_phi_1D,
                                 py_matrix.ones((num_samples, len(probs))))
    py_draws = py_rand.rand_softmax(py_phi)

    torch_phi_1D = torch_matrix.float_tensor(py_phi_1D)
    torch_phi = torch_matrix.broadcast(torch_phi_1D,
                                       torch_matrix.ones((num_samples, len(probs))))
    torch_draws = torch_rand.rand_softmax(torch_phi)

    py_counts = py_matrix.tsum(py_draws, axis=0)
    torch_counts = torch_matrix.tsum(torch_draws, axis=0)

    py_num_diff = py_matrix.tsum(0.5*py_func.tabs(py_counts - expected_counts))
    py_sigma_diff = py_num_diff / py_matrix.tsum(py_func.sqrt(expected_counts))
    assert py_sigma_diff < 2, \
        "python random softmax distribution appears inaccurate"

    torch_expected_counts = torch_matrix.float_tensor(expected_counts)
    torch_num_diff = torch_matrix.tsum(0.5*torch_func.tabs(torch_counts - torch_expected_counts))
    torch_sigma_diff = torch_num_diff / torch_matrix.tsum(torch_func.sqrt(torch_expected_counts))
    assert torch_sigma_diff < 2, \
        "torch random softmax distribution appears inaccurate"


def test_1d_conventions():
    A = py_matrix.float_tensor(np.arange(100))
    B = torch_matrix.float_tensor(A)

    def float_type(py_func, t_func, A, B):
        a = py_func(A, axis=0)
        b = t_func(B, axis=0)
        assert isinstance(a, np.float32)
        assert isinstance(b, float)
        a = py_func(A, axis=None)
        b = t_func(B, axis=None)
        assert isinstance(a, np.float32)
        assert isinstance(b, float)
        assert a == np.float32(b)

    py_funcs = [py_matrix.norm, py_matrix.tmax, py_matrix.tmin, py_matrix.mean,
               py_matrix.var, py_matrix.std, py_matrix.tsum, py_matrix.tprod]
    t_funcs = [torch_matrix.norm, torch_matrix.tmax, torch_matrix.tmin, torch_matrix.mean,
               torch_matrix.var, torch_matrix.std, torch_matrix.tsum, torch_matrix.tprod]

    for i in range(len(py_funcs)):
        float_type(py_funcs[i], t_funcs[i], A, B)

    def byte_type(py_func, t_func, A, B):
        a = py_func(A, axis=0)
        b = t_func(B, axis=0)
        assert isinstance(a, np.bool_)
        assert isinstance(b, int)
        a = py_func(A, axis=None)
        b = t_func(B, axis=None)
        assert isinstance(a, np.bool_)
        assert isinstance(b, int)
        assert bool(a) == bool(b)

    py_funcs = [py_matrix.tany, py_matrix.tall]
    t_funcs = [torch_matrix.tany, torch_matrix.tall]

    for i in range(len(py_funcs)):
        byte_type(py_funcs[i], t_funcs[i], A, B)

def test_do_nothing():
    t = py_matrix.float_tensor([1,2,3])
    T = [t,t,t]
    result_pre = be.do_nothing(T)
    result_ref = T
    assert isinstance(result_pre, list)
    assert isinstance(result_ref, list)
    for i,I in enumerate(result_ref):
        I == result_pre[i]

def test_force_list():
    t = py_matrix.float_tensor([1,2,3])
    assert isinstance(be.force_list(t), list)

if __name__ == "__main__":
    pytest.main([__file__])
