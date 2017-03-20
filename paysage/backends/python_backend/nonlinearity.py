import numpy
import numexpr as ne
from . import typedef as T

EPSILON = float(numpy.finfo(numpy.float32).eps)
LOG2 = 0.6931471805599453

def tabs(x: T.Tensor) -> T.Tensor:
    """
    Elementwise absolute value of a tensor.

    """
    return ne.evaluate('abs(x)')

def exp(x: T.Tensor) -> T.Tensor:
    """
    Elementwise exponential function of a tensor.

    """
    return ne.evaluate('exp(x)')

def log(x: T.Tensor) -> T.Tensor:
    """
    Elementwise natural logarithm of a tensor.

    """
    return ne.evaluate('log(x)')

def tanh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise hyperbolic tangent of a tensor.

    """
    return ne.evaluate('tanh(x)')

def expit(x: T.Tensor) -> T.Tensor:
    """
    Elementwise expit (a.k.a. logistic) function of a tensor.

    """
    return ne.evaluate('(1 + tanh(x/2))/2')

def reciprocal(x: T.Tensor) -> T.Tensor:
    """
    Elementwise inverse of a tensor.

    """
    return numpy.reciprocal(x)

def atanh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise inverse hyperbolic tangent of a tensor.

    """
    y = numpy.clip(x, a_min=EPSILON-1, a_max=1-EPSILON)
    return ne.evaluate('arctanh(y)')

def sqrt(x: T.Tensor) -> T.Tensor:
    """
    Elementwise square root of a tensor.

    """
    return ne.evaluate('sqrt(x)')

def square(x: T.Tensor) -> T.Tensor:
    """
    Elementwise square of a tensor.

    """
    return ne.evaluate('x**2')

def tpow(x: T.Tensor, a: T.Scalar) -> T.Tensor:
    """
    Elementwise power of a tensor x to power a.

    """
    return ne.evaluate('x**a')

def cosh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise hyperbolic cosine of a tensor.

    """
    return ne.evaluate('cosh(x)')

def logaddexp(x1: T.Tensor, x2: T.Tensor) -> T.Tensor:
    """
    Elementwise logaddexp function: log(exp(x1) + exp(x2))

    """
    return numpy.logaddexp(x1, x2)

def logcosh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise logarithm of the hyperbolic cosine of a tensor.

    """
    return -LOG2 + logaddexp(-x, x)

def acosh(x: T.Tensor) -> T.Tensor:
    """
    Elementwise inverse hyperbolic cosine of a tensor.

    """
    y = numpy.clip(x,1+EPSILON, numpy.inf)
    return ne.evaluate('arccosh(y)')

def logit(x: T.Tensor) -> T.Tensor:
    """
    Elementwise logit function of a tensor. Inverse of the expit function.

    """
    y = numpy.clip(x, a_min=EPSILON, a_max=1-EPSILON)
    return ne.evaluate('log(y/(1-y))')

def softplus(x: T.Tensor) -> T.Tensor:
    """
    Elementwise softplus function of a tensor.

    """
    return numpy.logaddexp(0, x)

def cos(x: T.Tensor) -> T.Tensor:
    """
    Elementwise cosine of a tensor.

    """
    return ne.evaluate('cos(x)')

def sin(x: T.Tensor) -> T.Tensor:
    """
    Elementwise sine of a tensor.

    """
    return ne.evaluate('sin(x)')
