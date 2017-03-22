import torch
from . import typedef as T

DEFAULT_SEED = 137

def set_seed(n: int = DEFAULT_SEED):
    """
    Set the seed of the random number generator.

    Notes:
        Default seed is 137.

    Args:
        n: Random seed.

    Returns:
        None

    """
    torch.manual_seed(int(n))

def rand(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Generate a tensor of the specified shape filled with uniform random numbers
    between 0 and 1.

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    return torch.rand(shape)

def randn(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Generate a tensor of the specified shape filled with random numbers
    drawn from a standard normal distribution (mean = 0, variance = 1).

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    return torch.randn(shape)
