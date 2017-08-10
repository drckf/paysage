import numpy as np

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



if __name__ == "__main__":
    pytest.main([__file__])
