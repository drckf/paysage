import torch, numpy
from . import matrix
from . import nonlinearity as nl
from . import typedef as T

device = matrix.device

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
    if device.type == 'cpu':
        torch.manual_seed(int(n))
    else:
        torch.cuda.manual_seed(int(n))

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

def rand_int(a: int, b: int, shape: T.Tuple[int]) -> T.LongTensor:
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
    return torch.randint(a,b, shape, device=device, dtype=T.Long)

def rand_samples(tensor: T.FloatTensor, num: int) -> T.FloatTensor:
    """
    Collect a random number samples from a tensor with replacement.
    Only supports the input tensor being a vector.

    Args:
        tensor ((num_samples)): a vector of values.
        num (int): the number of samples to take.

    Returns:
        samples ((num)): a vector of sampled values.

    """
    ix = rand_int(0, len(tensor), (num,))
    return tensor[ix]

def shuffle_(tensor: T.FloatTensor) -> None:
    """
    Shuffle the rows of a tensor.

    Notes:
        Modifies tensor in place.

    Args:
        tensor (shape): a tensor to shuffle.

    Returns:
        None

    """
    tensor[:] = tensor[torch.randperm(len(tensor), device=device, dtype=T.Long)]

def rand_softmax_units(phi: T.FloatTensor) -> T.FloatTensor:
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
    cum_probs = matrix.cumsum(probs, 1)
    ref_probs = rand((len(phi), 1))
    on_units = matrix.tsum(matrix.cast_long(cum_probs < ref_probs), axis=1,
                           keepdims=True)
    matrix.clip_(on_units, a_min=0, a_max=max_index)
    return on_units

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
    on_units = rand_softmax_units(phi)
    return matrix.zeros_like(phi).scatter_(1, on_units, 1)
