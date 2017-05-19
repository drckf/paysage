import numpy as np

from paysage import math_utils
from paysage import backends as be

import pytest

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
        mv.calculate(s[i*10000:(i+1)*10000])

    assert be.allclose(ref_mean, mv.mean)
    assert be.allclose(ref_var, mv.var)

if __name__ == "__main__":
    pytest.main([__file__])
