import numpy
from . import matrix
from . import nonlinearity as nl
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
    numpy.random.seed(int(n))

def rand(shape: T.Tuple[int]) -> T.Tensor:
    """
    Generate a tensor of the specified shape filled with uniform random numbers
    between 0 and 1.

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    return matrix.float_tensor(numpy.random.rand(*shape))

def rand_like(tensor: T.Tensor) -> T.Tensor:
    """
    Generate a tensor of the same shape as the specified tensor

    Args:
        tensor: tensor with desired shape.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    return matrix.float_tensor(numpy.random.rand(*matrix.shape(tensor)))

def randn(shape: T.Tuple[int]) -> T.Tensor:
    """
    Generate a tensor of the specified shape filled with random numbers
    drawn from a standard normal distribution (mean = 0, variance = 1).

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    return matrix.float_tensor(numpy.random.randn(*shape))

def randn_like(tensor: T.Tensor) -> T.Tensor:
    """
    Generate a tensor of the same shape as the specified tensor
    filled with normal(0,1) random numbers

    Args:
        tensor: tensor with desired shape.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    return matrix.float_tensor(numpy.random.randn(*matrix.shape(tensor)))

def rand_softmax(phi: T.Tensor) -> T.Tensor:
    """
    Draw random 1-hot samples according to softmax probabilities.

    Given an effective field vector v,
    the softmax probabilities are p = exp(v) / sum(exp(v))

    A 1-hot vector x is sampled according to p.

    Args:
        phi (tensor (batch_size, num_units)): the effective field

    Returns:
        tensor (batch_size, num_units): random 1-hot samples
            from the softmax distribution.

    """
    max_index = matrix.shape(phi)[1]-1
    probs = nl.softmax(phi)
    cum_probs = numpy.cumsum(probs, axis=1)
    ref_probs = numpy.random.rand(len(phi), 1)
    on_units = matrix.tsum(cum_probs < ref_probs, axis=1)
    matrix.clip_inplace(on_units, a_min=0, a_max=max_index)
    units = matrix.zeros_like(phi)
    units[range(len(phi)), on_units] = 1.
    return matrix.float_tensor(units)
