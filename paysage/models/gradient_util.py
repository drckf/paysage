from cytoolz import compose, partial
from math import sqrt
from collections import namedtuple

from .. import backends as be

Gradient = namedtuple("Gradient", [
    "layers", # List[List[ParamsLayer]]
    "weights" # List[List[ParamsWeights]]
])

"""
Utility functions for manipulating Gradient objects
"""

def null_grad(model):
    """
    Return a gradient object filled with empty lists.

    Args:
        model: a BoltzmannMachine object

    Returns:
        Gradient

    """
    return Gradient([[] for layer in model.layers], [[] for conn in model.connections])

def zero_grad(model):
    """
    Return a gradient object filled with zero tensors.

    Args:
        model: a BoltzmannMachine object

    Returns:
        Gradient

    """
    return Gradient(
        [layer.zero_derivatives() for layer in model.layers],
        [conn.weights.zero_derivatives() for conn in model.connections]
        )

def random_grad(model):
    """
    Return a gradient object filled with random numbers.

    Args:
        model: a BoltzmannMachine object

    Returns:
        Gradient

    """
    return Gradient(
        [layer.random_derivatives() for layer in model.layers],
        [conn.weights.random_derivatives() for conn in model.connections]
        )

def grad_accumulate(func, grad):
    """
    Apply a function entrywise over a Gradient object,
    accumulating the result.

    Args:
        func (callable): function with one argument
        grad (Gradient)

    returns:
        float

    """
    result = 0
    for layer in grad.layers:
        for sub_layer in layer:
            result += be.accumulate(func, sub_layer)
    for weight in grad.weights:
        for sub_weight in weight:
            result += be.accumulate(func, sub_weight)
    return result

def grad_apply(func, grad):
    """
    Apply a function entrywise over a Gradient object.

    Args:
        func (callable)
        grad (Gradient)

    Returns:
        Gradient

    """
    return Gradient(
        [[be.apply(func, sub_layer) for sub_layer in layer] for layer in grad.layers],
        [[be.apply(func, sub_weight) for sub_weight in weight] for weight in grad.weights]
    )

def grad_apply_(func_, grad):
    """
    Apply a function entrywise over a Gradient object.

    Notes:
        Modifies elements of grad in place.

    Args:
        func_ (callable, in place operation)
        grad (Gradient)

    Returns:
        None

    """
    for layer in grad.layers:
        for sub_layer in layer:
            be.apply_(func_, sub_layer)
    for weight in grad.weights:
        for sub_weight in weight:
            be.apply_(func_, sub_weight)

def grad_mapzip(func, grad1, grad2):
    """
    Apply a function entrywise over the zip of two Gradient objects.

    Args:
        func_ (callable, in place operation)
        grad (Gradient)

    Returns:
        Gradient

    """
    n = len(grad1.layers)
    m = len(grad1.weights)
    return Gradient(
        [[be.mapzip(func, z[0], z[1]) for z in zip(grad1.layers[i], grad2.layers[i])]
        for i in range(n)],
        [[be.mapzip(func, z[0], z[1]) for z in zip(grad1.weights[i], grad2.weights[i])]
        for i in range(m)]
        )

def grad_mapzip_(func_, grad1, grad2):
    """
    Apply an in place function entrywise over the zip of two Gradient objects.

    Notes:
        Modifies elements of grad1 in place.

    Args:
        func_ (callable, in place operation)
        grad1 (Gradient)
        grad2 (Gradient)

    Returns:
        None

    """
    n = len(grad1.layers)
    m = len(grad1.weights)
    for i in range(n):
        for z in zip(grad1.layers[i], grad2.layers[i]):
            be.mapzip_(func_, z[0], z[1])
    for j in range(m):
        for z in zip(grad1.weights[j], grad2.weights[j]):
            be.mapzip_(func_, z[0], z[1])

def grad_norm(grad):
    """
    Compute the l2 norm of the gradient.

    Args:
        grad (Gradient)

    Returns:
        magnitude (float)

    """
    tensor_sum_square = compose(be.tsum, be.square)
    return sqrt(grad_accumulate(tensor_sum_square, grad))

def grad_normalize_(grad):
    """
    Normalize the gradient vector with respect to the L2 norm

    Args:
        grad (Gradient)

    Return:
        None
    """
    nrm = grad_norm(grad)
    grad_apply_(partial(be.tmul_, be.float_scalar(1.0/nrm)), grad)

def grad_rms(grad):
    """
    Compute the root-mean-square of the gradient.

    Args:
        grad (Gradient)

    Returns:
        rms (float)

    """
    n = len(grad.layers) + len(grad.weights)
    tensor_mean_square = compose(be.mean, be.square)
    return sqrt(grad_accumulate(tensor_mean_square, grad) / n)

def grad_flatten(grad):
    """
    Returns a flat vector of gradient parameters

    Args:
        grad (Gradient)

    Returns:
        (tensor): vectorized gradient
    """
    v = []
    for l in grad.layers:
        for c in l:
            v.extend([be.flatten(i) for i in c])
    for w in grad.weights:
        for c in w:
            v.extend([be.flatten(i) for i in c])
    return be.hstack(v)
