import numpy
import numexpr as ne

EPSILON = float(numpy.finfo(numpy.float32).eps)
LOG2 = 0.6931471805599453

def tabs(x):
    """
    Elementwise absolute value of a tensor.

    """
    return ne.evaluate('abs(x)')

def exp(x):
    """
    Elementwise exponential function of a tensor.

    """
    return ne.evaluate('exp(x)')

def log(x):
    """
    Elementwise natural logarithm of a tensor.

    """
    z = numpy.clip(x, a_min=EPSILON, a_max=numpy.inf)
    return ne.evaluate('log(z)')

def tanh(x):
    """
    Elementwise hyperbolic tangent of a tensor.

    """
    return ne.evaluate('tanh(x)')

def expit(x):
    """
    Elementwise expit (a.k.a. logistic) function of a tensor.

    """
    return ne.evaluate('(1 + tanh(x/2))/2')

def reciprocal(x):
    """
    Elementwise inverse of a tensor.

    """
    return numpy.reciprocal(x)

def atanh(x):
    """
    Elementwise inverse hyperbolic tangent of a tensor.

    """
    y = numpy.clip(x, a_min=EPSILON-1, a_max=1-EPSILON)
    return ne.evaluate('arctanh(y)')

def sqrt(x):
    """
    Elementwise square root of a tensor.

    """
    return ne.evaluate('sqrt(x)')

def square(x):
    """
    Elementwise square of a tensor.

    """
    return ne.evaluate('x**2')

def tpow(x, a):
    """
    Elementwise power of a tensor x to power a.

    """
    return ne.evaluate('x**a')

def cosh(x):
    """
    Elementwise hyperbolic cosine of a tensor.

    """
    return ne.evaluate('cosh(x)')

def logaddexp(x1, x2):
    """
    Elementwise logaddexp function: log(exp(x1) + exp(x2))

    """
    return numpy.logaddexp(x1, x2)

def logcosh(x):
    """
    Elementwise logarithm of the hyperbolic cosine of a tensor.

    """
    return -LOG2 + logaddexp(-x, x)

def acosh(x):
    """
    Elementwise inverse hyperbolic cosine of a tensor.

    """
    y = numpy.clip(x,1+EPSILON, numpy.inf)
    return ne.evaluate('arccosh(y)')

def logit(x):
    """
    Elementwise logit function of a tensor. Inverse of the expit function.

    """
    y = numpy.clip(x, a_min=EPSILON, a_max=1-EPSILON)
    return ne.evaluate('log(y/(1-y))')

def softplus(x):
    """
    Elementwise softplus function of a tensor.

    """
    return numpy.logaddexp(0, x)

def cos(x):
    """
    Elementwise cosine of a tensor.

    """
    return ne.evaluate('cos(x)')

def sin(x):
    """
    Elementwise sine of a tensor.

    """
    return ne.evaluate('sin(x)')
