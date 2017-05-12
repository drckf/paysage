from .matrix import float_tensor
import numpy.random as random
from . import typedef as T

DEFAULT_SEED = 137

def set_seed(n: int = DEFAULT_SEED) -> None:
    """
    Set the seed of the random number generator.

    Notes:
        Default seed is 137.

    Args:
        n: Random seed.

    Returns:
        None

    """
    random.seed(int(n))

def rand(shape: T.Tuple[int]) -> T.Tensor:
    """
    Generate a tensor of the specified shape filled with uniform random numbers
    between 0 and 1.

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    return float_tensor(random.rand(*shape))

def rand_like(tensor: T.Tensor) -> T.Tensor:
    """
    Generate a tensor of the same shape as the specified tensor

    Args:
        tensor: tensor with desired shape.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    return float_tensor(random.rand(*tensor.shape))

def randn(shape: T.Tuple[int]) -> T.Tensor:
    """
    Generate a tensor of the specified shape filled with random numbers
    drawn from a standard normal distribution (mean = 0, variance = 1).

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    return float_tensor(random.randn(*shape))
