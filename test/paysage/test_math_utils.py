import numpy as np
import math

from paysage import math_utils
from paysage import backends as be

import pytest

# ----- MeanCalculator ----- #

def test_mean():
    # create some random data
    num = 100000
    num_steps = 10
    stepsize = num // num_steps
    s = be.rand((num,))

    # reference result
    ref_mean = be.mean(s)

    # do the online calculation
    mv = math_utils.MeanCalculator()
    for i in range(num_steps):
        mv.update(s[i*stepsize:(i+1)*stepsize])

    assert be.allclose(be.float_tensor(np.array([ref_mean])),
                       be.float_tensor(np.array([mv.mean])))



# ----- MeanArrayCalculator ----- #

def test_mean_2d():
    # create some random data
    num =5000
    num_steps = 10
    stepsize = num // num_steps
    s = be.rand((num,10))

    # reference result
    ref_mean = be.mean(s, axis=0)

    # do the online calculation
    mv = math_utils.MeanArrayCalculator()
    for i in range(num_steps):
        mv.update(s[i*stepsize:(i+1)*stepsize], axis=0)

    assert be.allclose(ref_mean, mv.mean)


# ----- MeanVarianceCalculator ----- #

def test_mean_variance():
    # create some random data
    num = 100000
    num_steps = 10
    stepsize = num // num_steps
    s = be.rand((num,))

    # reference result
    ref_mean = be.mean(s)
    ref_var = be.var(s)

    # do the online calculation
    mv = math_utils.MeanVarianceCalculator()
    for i in range(num_steps):
        mv.update(s[i*stepsize:(i+1)*stepsize])

    assert be.allclose(be.float_tensor(np.array([ref_mean])),
                       be.float_tensor(np.array([mv.mean])))
    assert be.allclose(be.float_tensor(np.array([ref_var])),
                       be.float_tensor(np.array([mv.var])),
                       rtol=1e-3, atol=1e-5)


# ----- MeanVarianceArrayCalculator ----- #


def test_mean_variance_2d():
    # create some random data
    num = 10000
    dim2 = 10
    num_steps = 10
    stepsize = num // num_steps
    s = be.rand((num,dim2))

    # reference result
    ref_mean = be.mean(s, axis=0)
    ref_var = be.var(s, axis=0)

    # do the online calculation
    mv = math_utils.MeanVarianceArrayCalculator()
    for i in range(num_steps):
        mv.update(s[i*stepsize:(i+1)*stepsize])

    assert be.allclose(ref_mean, mv.mean)
    assert be.allclose(ref_var, mv.var, rtol=1e-3, atol=1e-5)


def test_mean_variance_serialization():
    # create some random data
    num = 100
    dim2 = 10
    num_steps = 10
    stepsize = num // num_steps
    s = be.rand((num, dim2))

    # do the online calculation
    mv = math_utils.MeanVarianceArrayCalculator()
    for i in range(num_steps):
        mv.update(s[i*stepsize:(i+1)*stepsize])

    df = mv.to_dataframe()
    mv_serial = math_utils.MeanVarianceArrayCalculator.from_dataframe(df)

    assert be.allclose(mv_serial.mean, mv.mean)
    assert be.allclose(mv_serial.var, mv.var)
    assert be.allclose(mv_serial.square, mv.square)
    assert mv_serial.num == mv.num

def test_pdist():
    n=500
    a_shape = (1000, n)
    b_shape = (1000, n)

    # distance distributions
    a_mean, a_scale = 1, 1
    b_mean, b_scale = -1, 1

    be.set_seed()
    a = a_mean + a_scale * be.randn(a_shape)
    b = b_mean + b_scale * be.randn(b_shape)

    dists = math_utils.pdist(a, b)
    dists_t = math_utils.pdist(b, a)
    assert be.shape(dists) == (1000,1000)
    assert be.allclose(be.transpose(dists_t), dists)
    assert be.mean(dists) > 2*math.sqrt(n) and be.mean(dists) < 3*math.sqrt(n)

def test_find_k_nearest_neighbors():
    n=20
    shp = (20, n)

    perm = be.rand_int(0,20,(20,))
    k = 1
    be.set_seed()

    y = be.randn(shp)
    x = y[perm]

    indices, _distances = math_utils.find_k_nearest_neighbors(x, y, k)

    assert be.allclose(indices, perm)
    assert be.allclose(_distances, be.zeros((20,)), 1e-2, 1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
