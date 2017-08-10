import torch, numpy
from . import matrix
from . import nonlinearity as nl
from . import typedef as T

DTYPE=matrix.DTYPE

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
    # set the seed for the cpu generator
    torch.manual_seed(int(n))
    numpy.random.seed(int(n))
    # set the seed for the gpu generator if needed
    DTYPE.manual_seed(int(n))

def rand(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Generate a tensor of the specified shape filled with uniform random numbers
    between 0 and 1.

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    x = matrix.zeros(shape)
    x.uniform_()
    return x

def rand_like(tensor: T.TorchTensor) -> T.FloatTensor:
    """
    Generate a tensor of the same shape as the specified tensor
    filled with uniform [0,1] random numbers

    Args:
        tensor: tensor with desired shape.

    Returns:
        tensor: Random numbers between 0 and 1.

    """
    x = matrix.zeros_like(tensor)
    x.uniform_()
    return x

def randn(shape: T.Tuple[int]) -> T.FloatTensor:
    """
    Generate a tensor of the specified shape filled with random numbers
    drawn from a standard normal distribution (mean = 0, variance = 1).

    Args:
        shape: Desired shape of the random tensor.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    x = matrix.zeros(shape)
    x.normal_()
    return x

def randn_like(tensor: T.FloatTensor) -> T.FloatTensor:
    """
    Generate a tensor of the same shape as the specified tensor
    filled with normal(0,1) random numbers

    Args:
        tensor: tensor with desired shape.

    Returns:
        tensor: Random numbers between from a standard normal distribution.

    """
    x = matrix.zeros_like(tensor)
    x.normal_()
    return x

def rand_softmax(phi: T.FloatTensor) -> T.FloatTensor:
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
    cum_probs = torch.cumsum(probs, 1)
    ref_probs = rand((len(phi), 1))
    on_units = matrix.int_tensor(matrix.tsum(cum_probs < ref_probs, axis=1, keepdims=True))
    matrix.clip_inplace(on_units, a_min=0, a_max=max_index)
    return matrix.zeros_like(phi).scatter_(1, on_units, 1)
