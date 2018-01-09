from . import backends as be
from cytoolz import partial, compose

def do_nothing(tensor):
    """
    Identity function.

    Args:
        Anything.

    Returns:
        Anything.

    """
    return tensor

def scale(tensor, denominator):
    """
    Rescale the values in a tensor by the denominator.

    Args:
        tensor: A tensor.
        denominator (float)

    Returns:
        float tensor

    """
    return tensor/denominator

def l2_normalize(tensor):
    """
    Divide the rows of the tensory by their L2 norms.

    Args:
        tensor (num_samples, num_units)

    Returns:
        tensor (num_samples, num_units)

    """
    norm = be.norm(tensor, axis=1, keepdims=True)
    return be.divide(norm, tensor)

def l1_normalize(tensor):
    """
    Divide the rows of the tensor by their L1 norms.

    Args:
        tensor (num_samples, num_units)

    Returns:
        tensor (num_samples, num_units)

    """
    norm = be.tsum(tensor, axis=1, keepdims=True)
    return be.divide(norm, tensor)

def binarize_color(tensor):
    """
    Scales an int8 "color" value to [0, 1].

    Args:
        tensor

    Returns:
        float tensor

    """
    return be.float_tensor(be.tround(tensor/255))

def binary_to_ising(tensor):
    """
    Scales a [0, 1] value to [-1, 1].

    Args:
        tensor

    Returns:
        float tensor

    """
    return 2.0 * tensor - 1.0

def color_to_ising(tensor):
    """
    Scales an int8 "color" value to [-1, 1].

    Args:
        tensor

    Returns:
        float tensor

    """
    return binary_to_ising(binarize_color(tensor))

def one_hot(data, category_list):
    """
    Convert a categorical variable into a one-hot code.

    Args:
        data (tensor (num_samples, 1)): a column of the data matrix that is categorical
        category_list: the list of categories

    Returns:
        one-hot encoded data (tensor (num_samples, num_categories))

    """
    units = be.zeros((len(data), len(category_list)))
    on_units = be.int_tensor(list(map(category_list.index, be.flatten(data))))
    be.scatter_(units, on_units, 1.)
    return units
