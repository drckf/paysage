from . import backends as be


def non_negative(tensor):
    """
    Set any negative entries of the input tensor to zero.

    Notes:
        Modifies the input tensor in place!

    Args:
        tensor

    Returns:
        None

    """
    be.clip_(tensor, a_min=0.0)


def non_positive(tensor):
    """
    Set any positive entries of the input tensor to zero.

    Notes:
        Modifies the input tensor in place!

    Args:
        tensor

    Returns:
        None

    """
    be.clip_(tensor, a_max=0.0)


def diagonal(tensor):
    """
    Set any off-diagonal entries of the input tensor to zero.

    Notes:
        Modifies the input tensor in place!

    Args:
        tensor

    Returns:
        None

    """
    tensor[:] = be.diagonal_matrix(be.diag(tensor))


def zero_row(tensor, index):
    """
    Set any entries of in the given row of the input tensor to zero.

    Notes:
        Modifies the input tensor in place!

    Args:
        tensor
        index (int): index of the row to set to zero

    Returns:
        None

    """
    tensor[index, :] = 0.0


def zero_column(tensor, index):
    """
    Set any entries of in the given column of the input tensor to zero.

    Notes:
        Modifies the input tensor in place!

    Args:
        tensor
        index (int): index of the column to set to zero

    Returns:
        None

    """
    tensor[:, index] = 0.0


def zero_mask(tensor, mask):
    """
    Set the given entries of the input tensor to zero.

    Notes:
        Modifies the input tensor in place!

    Args:
        tensor
        mask: a binary mask of the same shape as tensor. entries where the mask
            is 1 will be set to zero

    Returns:
        None

    """
    tensor[mask] = 0.0


def fixed_column_norm(tensor):
    """
    Renormalize the tensor so that all of its columns have the same norm.

    Notes:
        Modifies the input tensor in place!

    Args:
        tensor

    Returns:
        None

    """
    norms = be.norm(tensor, axis=0)
    be.divide_(norms / be.mean(norms), tensor)
