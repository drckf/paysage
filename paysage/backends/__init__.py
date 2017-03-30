import json, os


# load the configuration file with the backend specification
filedir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(filedir,"config.json"), "r") as infile:
    config = json.load(infile)


# import the functions from the specified backend
if config['backend'] == 'python':
    from .python_backend.matrix import *
    from .python_backend.nonlinearity import *
    from .python_backend.rand import *
elif config['backend'] == 'pytorch':
    from .pytorch_backend.matrix import *
    from .pytorch_backend.nonlinearity import *
    from .pytorch_backend.rand import *
else:
    raise ValueError(
    "Unknown backend {}".format(config['backend'])
    )


# ----- COMMON FUNCTIONALITY ----- #

from collections import namedtuple

def add_tuples(a, b):
    """
    Add tuple a to tuple b entrywise.

    Args:
        a (namedtuple): (key: tensor)
        b (namedtuple): (key: tensor)

    Returns:
        namedtuple: a + b (key: tensor)

    """
    return type(a)(*(add(x[0], x[1]) for x in zip(a, b)))

def subtract_tuples(a, b):
    """
    Subtract tuple a from tuple b entrywise.

    Args:
        a (namedtuple): (key: tensor)
        b (namedtuple): (key: tensor)

    Returns:
        namedtuple: b - a (key: tensor)

    """
    return type(a)(*(subtract(x[0], x[1]) for x in zip(a, b)))

def multiply_tuples(a, b):
    """
    Multiply tuple b by tuple a entrywise.

    Args:
        a (namedtuple): (key: tensor)
        b (namedtuple): (key: tensor)

    Returns:
        namedtuple: a * b (key: tensor)

    """
    return type(a)(*(multiply(x[0], x[1]) for x in zip(a, b)))

def divide_tuples(a, b):
    """
    Divide tuple b by tuple a entrywise.

    Args:
        a (namedtuple; non-zero): (key: tensor)
        b (namedtuple): (key: tensor)

    Returns:
        namedtuple: b / a (key: tensor)

    """
    return type(a)(*(divide(x[0], x[1]) for x in zip(a, b)))
