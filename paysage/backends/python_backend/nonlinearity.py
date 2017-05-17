import numpy
import numexpr as ne
from . import typedef as T

EPSILON = float(numpy.finfo(numpy.float32).eps)
LOG2 = 0.6931471805599453

def pi() -> T.Scalar:
    return T.Scalar(math.pi)

def tmul(a: T.Scalar, x: T.Tensor) -> T.Tensor:
    """
    Elementwise multiplication of tensor x by scalar a.

    Args:
        x: A tensor.
        a: scalar.

    Returns:
        tensor: Elementwise a * x.

    """
    return a * x

def tmul_(a: T.Scalar, x: T.Tensor) -> T.Tensor:
    """
    Elementwise multiplication of tensor x by scalar a.

    Notes:
        Modifes x in place

    Args:
        x: A tensor.
        a: scalar.

    Returns:
        tensor: Elementwise a * x.

    """
    ne.evaluate('a * x', out=x)

def tabs(x: T.Tensor) -> T.Tensor:
    """
    Elementwise absolute value of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor (non-negative): Absolute value of x.

    """
    return ne.evaluate('abs(x)')

def exp(x: T.Tensor) -> T.Tensor:
    """
    Elementwise exponential function of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor (non-negative): Elementwise exponential.

    """
    return ne.evaluate('exp(x)')

def log(x: T.Tensor) -> T.Tensor:
    """
    Elementwise natural logarithm of a tensor.

    Args:
        x (non-negative): A tensor.

    Returns:
        tensor: Elementwise natural logarithm.

    """
    z = numpy.clip(x, a_min=EPSILON, a_max=numpy.inf)
    return ne.evaluate('log(z)')

def tanh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise hyperbolic tangent of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise hyperbolic tangent.

    """
    return ne.evaluate('tanh(x)')

def expit(x: T.Tensor) -> T.Tensor:
    """
    Elementwise expit (a.k.a. logistic) function of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise expit (a.k.a. logistic).

    """
    return ne.evaluate('(1 + tanh(x/2))/2')

def reciprocal(x: T.Tensor) -> T.Tensor:
    """
    Elementwise inverse of a tensor.

    Args:
        x (non-zero): A tensor:

    Returns:
        tensor: Elementwise inverse.

    """
    return numpy.reciprocal(x)

def atanh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise inverse hyperbolic tangent of a tensor.

    Args:
        x (between -1 and +1): A tensor.

    Returns:
        tensor: Elementwise inverse hyperbolic tangent

    """
    y = numpy.clip(x, a_min=EPSILON-1, a_max=1-EPSILON)
    return ne.evaluate('arctanh(y)')

def sqrt(x: T.Tensor) -> T.Tensor:
    """
    Elementwise square root of a tensor.

    Args:
        x (non-negative): A tensor.

    Returns:
        tensor(non-negative): Elementwise square root.

    """
    return ne.evaluate('sqrt(x)')

def square(x: T.Tensor) -> T.Tensor:
    """
    Elementwise square of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor (non-negative): Elementwise square.

    """
    return ne.evaluate('x**2')

def tpow(x: T.Tensor, a: float) -> T.Tensor:
    """
    Elementwise power of a tensor x to power a.

    Args:
        x: A tensor.
        a: Power.

    Returns:
        tensor: Elementwise x to the power of a.

    """
    return ne.evaluate('x**a')

def cosh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise hyperbolic cosine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise hyperbolic cosine.

    """
    return ne.evaluate('cosh(x)')

def logaddexp(x1: T.Tensor, x2: T.Tensor) -> T.Tensor:
    """
    Elementwise logaddexp function: log(exp(x1) + exp(x2))

    Args:
        x1: A tensor.
        x2: A tensor.

    Returns:
        tensor: Elementwise logaddexp.

    """
    return numpy.logaddexp(x1, x2)

def logcosh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise logarithm of the hyperbolic cosine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise logarithm of the hyperbolic cosine.

    """
    return -LOG2 + logaddexp(-x, x)

def acosh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise inverse hyperbolic cosine of a tensor.

    Args:
        x (greater than 1): A tensor.

    Returns:
        tensor: Elementwise inverse hyperbolic cosine.

    """
    y = numpy.clip(x,1+EPSILON, numpy.inf)
    return ne.evaluate('arccosh(y)')

def logit(x: T.Tensor) -> T.Tensor:
    """
    Elementwise logit function of a tensor. Inverse of the expit function.

    Args:
        x (between 0 and 1): A tensor.

    Returns:
        tensor: Elementwise logit function

    """
    y = numpy.clip(x, a_min=EPSILON, a_max=1-EPSILON)
    return ne.evaluate('log(y/(1-y))')

def softplus(x: T.Tensor) -> T.Tensor:
    """
    Elementwise softplus function of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise softplus.

    """
    return numpy.logaddexp(0, x)

def cos(x: T.Tensor) -> T.Tensor:
    """
    Elementwise cosine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise cosine.

    """
    return ne.evaluate('cos(x)')

def sin(x: T.Tensor) -> T.Tensor:
    """
    Elementwise sine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise sine.

    """
    return ne.evaluate('sin(x)')
