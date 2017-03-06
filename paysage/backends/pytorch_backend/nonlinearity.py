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

def logaddexp(x1, x2):
    # log(exp(x1) + exp(x2))
    # = log( exp(x1) (1 + exp(x2 - x1))) = x1 + log(1 + exp(x2 - x1))
    # = log( exp(x2) (exp(x1 - x2) + 1)) = x2 + log(1 + exp(x1 - x2))
    diff = torch.min(x2 - x1, x1 - x2)
    return torch.max(x1, x2) + torch.log1p(exp(diff))

def logcosh(x):
    return -LOG2 + logaddexp(-x, x)

def acosh(x):
    y = matrix.clip(x, a_min=EPSILON, a_max = 1 - EPSILON)
    return sqrt((y-1)/(1-y)) * torch.acos(x)

def logit(x):
    y = matrix.clip(x, a_min=EPSILON, a_max = 1 - EPSILON)
    return torch.log(y / (1 - y))

def softplus(x):
    return torch.log1p(exp(x))

def cos(x):
    return torch.cos(x)

def sin(x):
    return torch.sin(x)
