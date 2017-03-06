from .matrix import float_tensor
import numpy.random as random

DEFAULT_SEED = 137

def set_seed(n=DEFAULT_SEED):
    """
    Set the seed of the random number generator.
    Default seed is 137.

    """
    random.seed(int(n))

def rand(shape):
    """
    Generate a tensor of the specified shape filled with uniform random numbers
    between 0 and 1.

    """
    return float_tensor(random.rand(*shape))

def randn(shape):
    """
    Generate a tensor of the specified shape filled with random numbers
    drawn from a standard normal distribution (mean = 0, variance = 1).

    """
    return float_tensor(random.randn(*shape))
