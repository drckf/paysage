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
    return matrix.cast_float(numpy.random.rand(*shape))

def rand_like(tensor: T.Tensor) -> T.Tensor:
    """
    Generate a tensor of the same shape as the specified tensor

    Args:
        tensor: tensor with desired shape.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    return numpy.random.rand(*matrix.shape(tensor)).astype(tensor.dtype)

def randn(shape: T.Tuple[int]) -> T.Tensor:
    """
    Generate a tensor of the specified shape filled with random numbers
    drawn from a standard normal distribution (mean = 0, variance = 1).

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    return numpy.random.randn(*shape).astype(T.Float)

def randn_like(tensor: T.Tensor) -> T.Tensor:
    """
    Generate a tensor of the same shape as the specified tensor
    filled with normal(0,1) random numbers

    Args:
        tensor: tensor with desired shape.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    return numpy.random.randn(*matrix.shape(tensor)).astype(tensor.dtype)

def rand_int(a: int, b: int, shape: T.Tuple[int]) -> T.Tensor:
    """
    Generate random integers in [a, b).
    Fills a tensor of a given shape

    Args:
        a (int): the minimum (inclusive) of the range.
        b (int): the maximum (exclusive) of the range.
        shape: the shape of the output tensor.

    Returns:
        tensor (shape): the random integer samples.

    """
    return numpy.random.randint(a, b, shape).astype(T.Long)

def rand_samples(tensor: T.Tensor, num: int) -> T.Tensor:
    """
    Collect a random number samples from a tensor with replacement.
    Only supports the input tensor being a vector.

    Args:
        tensor ((num_samples)): a vector of values.
        num (int): the number of samples to take.

    Returns:
        samples ((num)): a vector of sampled values.

    """
    ix = rand_int(0, len(tensor), (num))
    return tensor[ix]

def shuffle_(tensor: T.Tensor) -> None:
    """
    Shuffle the rows of a tensor.

    Notes:
        Modifies tensor in place.

    Args:
        tensor (shape): a tensor to shuffle.

    Returns:
        None

    """
    numpy.random.shuffle(tensor)

def rand_softmax_units(phi: T.Tensor) -> T.Tensor:
    """
    Draw random unit values according to softmax probabilities.

    Given an effective field vector v,
    the softmax probabilities are p = exp(v) / sum(exp(v))

    The unit values (the on-units for a 1-hot encoding)
    are sampled according to p.

    Args:
        phi (tensor (batch_size, num_units)): the effective field

    Returns:
        tensor (batch_size,): random unit values from the softmax distribution.

    """
    max_index = matrix.shape(phi)[1]-1
    probs = nl.softmax(phi)
    cum_probs = matrix.cumsum(probs, axis=1)
    ref_probs = rand((len(phi), 1))
    on_units = matrix.tsum(cum_probs < ref_probs, axis=1)
    matrix.clip_(on_units, a_min=0, a_max=max_index)
    return on_units

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
    on_units = rand_softmax_units(phi)
    units = matrix.zeros_like(phi)
    units[range(len(phi)), on_units] = 1.
    return units.astype(T.Float)
