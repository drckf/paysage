import numpy, torch
from . import matrix

EPSILON = float(numpy.finfo(numpy.float32).eps)
LOG2 = 0.6931471805599453

def tabs(x):
    """
    Elementwise absolute value of a tensor.

    """
    return torch.abs(x)

def exp(x):
    """
    Elementwise exponential function of a tensor.

    """
    return torch.exp(x)

def log(x):
    """
    Elementwise natural logarithm of a tensor.

    """
    return torch.log(x)

def tanh(x):
    """
    Elementwise hyperbolic tangent of a tensor.

    """
    return torch.tanh(x)

def expit(x):
    """
    Elementwise expit (a.k.a. logistic) function of a tensor.

    """
    return 0.5 * (1.0 + tanh(0.5 * x))

def reciprocal(x):
    """
    Elementwise inverse of a tensor.

    """
    return torch.reciprocal(x)

def atanh(x):
    """
    Elementwise inverse hyperbolic tangent of a tensor.

    """
    y = matrix.clip(x, a_min=EPSILON - 1, a_max=1 - EPSILON)
    return (log(1+y) - log(1-y)) / 2

def sqrt(x):
    """
    Elementwise square root of a tensor.

    """
    return torch.sqrt(x)

def square(x):
    """
    Elementwise square of a tensor.

    """
    return x * x

def tpow(x, a):
    """
    Elementwise power of a tensor x to power a.

    """
    return torch.pow(x, a)

def cosh(x):
    """
    Elementwise hyperbolic cosine of a tensor.

    """
    return torch.cosh(x)

def logaddexp(x1, x2):
    """
    Elementwise logaddexp function: log(exp(x1) + exp(x2))

    """
    # log(exp(x1) + exp(x2))
    # = log( exp(x1) (1 + exp(x2 - x1))) = x1 + log(1 + exp(x2 - x1))
    # = log( exp(x2) (exp(x1 - x2) + 1)) = x2 + log(1 + exp(x1 - x2))
    diff = torch.min(x2 - x1, x1 - x2)
    return torch.max(x1, x2) + torch.log1p(exp(diff))

def logcosh(x):
    """
    Elementwise logarithm of the hyperbolic cosine of a tensor.

    """
    return -LOG2 + logaddexp(-x, x)

def acosh(x):
    """
    Elementwise inverse hyperbolic cosine of a tensor.

    """
    y = matrix.clip(x, a_min=1+EPSILON)
    return log(y + sqrt(y+1) * sqrt(y-1))
    #return sqrt((y-1)/(1-y)) * torch.acos(x)

def logit(x):
    """
    Elementwise logit function of a tensor. Inverse of the expit function.

    """
    y = matrix.clip(x, a_min=EPSILON, a_max = 1 - EPSILON)
    return torch.log(y / (1 - y))

def softplus(x):
    """
    Elementwise softplus function of a tensor.

    """
    return torch.log1p(exp(x))

def cos(x):
    """
    Elementwise cosine of a tensor.

    """
    return torch.cos(x)

def sin(x):
    """
    Elementwise sine of a tensor.

    """
    return torch.sin(x)
