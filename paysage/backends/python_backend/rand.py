from . import matrix
import numpy.random as random

DEFAULT_SEED = 137

def set_seed(n=DEFAULT_SEED):
    random.seed(int(n))

def rand(shape):
    return matrix.float_tensor(random.rand(*shape))

def randn(shape):
    return matrix.float_tensor(random.randn(*shape))
