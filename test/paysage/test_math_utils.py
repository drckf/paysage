import numpy as np

from paysage import math_utils
from paysage import backends as be

import pytest

# ----- MeanCalculator ----- #

def test_mean():
    # create some random data
    s = be.rand((100000,))

    # reference result
    ref_mean = be.mean(s)

    # do the online calculation
    mv = math_utils.MeanVarianceCalculator()
    for i in range(10):
        mv.update(s[i*10000:(i+1)*10000])

    assert be.allclose(be.float_tensor(np.array([ref_mean])),
                       be.float_tensor(np.array([mv.mean])))


# ----- MeanVarianceCalculator ----- #

def test_mean_variance():
    # create some random data
    s = be.rand((100000,))

    # reference result
    ref_mean = be.mean(s)
    ref_var = be.var(s)

    # do the online calculation
    mv = math_utils.MeanVarianceCalculator()
    for i in range(10):
        mv.update(s[i*10000:(i+1)*10000])

    assert be.allclose(be.float_tensor(np.array([ref_mean])),
                       be.float_tensor(np.array([mv.mean])))
    assert be.allclose(be.float_tensor(np.array([ref_var])),
                       be.float_tensor(np.array([mv.var])))

if __name__ == "__main__":
    pytest.main([__file__])
