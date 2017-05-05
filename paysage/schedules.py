from . import backends as be

def constant(initial=1.0):
    """
    Constant i.e. x(t) = x.

    Args:
        initial (float)

    Returns:
        generator

    """
    value = initial
    while value > 0:
        yield value

def exponential_decay(initial=1.0, coefficient=0.9):
    """
    Exponential decay with coefficient alpha, i.e. x(t) = alpha^t.
    Sets x(0) = 1 and uses the recursive formula x(t+1) = alpha * x(t).

    Args:
        initial (float)
        coefficient (float in [0,1])

    Returns:
        generator

    """
    value = initial
    while value > 0:
        yield be.float_scalar(value)
        value *= coefficient

def power_lay_decay(initial=1.0, coefficient = 0.1):
    """
    Power law decay with coefficient alpha, i.e. x(t) = 1 / (1 + alpha * t).
    Sets x(0) = 1 and uses the recursive formula 1/x(t+1) = alpha + 1/x(t).

    Args:
        initial (float)
        coefficient (float in [0,1])

    Returns:
        generator

    """
    value = initial
    reciprocal = 1 / value
    while value > 0:
        yield be.float_scalar(value)
        reciprocal += coefficient
        value = 1 / reciprocal
