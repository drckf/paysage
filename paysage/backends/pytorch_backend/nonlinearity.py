import numpy, torch
from . import matrix

EPSILON = numpy.finfo(numpy.float32).eps
LOG2 = 0.6931471805599453

def tabs(x):
    return torch.abs(x)

def exp(x):
    return torch.exp(x)

def log(x):
    return torch.log(x)

def tanh(x):
    return torch.tanh(x)

def expit(x):
    return 0.5 * (1.0 + tanh(0.5 * x))

def reciprocal(x):
    return torch.reciprocal(x)

def atanh(x):
    y = matrix.clip(x, a_min=EPSILON, a_max = 1 - EPSILON)
    return 0.5 * (log(1+y) - log(1-y))

def sqrt(x):
    return torch.sqrt(x)

def square(x):
    return x * x

def tpow(x, a):
    return torch.pow(x, a)

def cosh(x):
    return torch.cosh

def logcosh(x):
    raise NotImplementedError

def acosh(x):
    raise NotImplementedError

def logit(x):
    raise NotImplementedError

def softplus(x):
    raise NotImplementedError

def cos(x):
    return torch.cos(s)

def sin(x):
    return torch.sin(x)
