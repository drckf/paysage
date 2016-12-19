import numpy, math
from numba import jit, vectorize
import numexpr as ne

EPSILON = numpy.finfo(numpy.float32).eps
LOG2 = 0.6931471805599453

def exp(x):
    return ne.evaluate('exp(x)')
    
def log(x):
    return ne.evaluate('log(x)')
    
def tanh(x):
    return ne.evaluate('tanh(x)')
    
def expit(x):
    return ne.evaluate('(1 + tanh(x/2))/2')
    
def atanh(x):
    y = numpy.clip(x, a_min=EPSILON-1, a_max=1-EPSILON)
    return ne.evaluate('arctanh(y)')
    
def sqrt(x):
    return ne.evaluate('sqrt(x)')
    
def cosh(x):
    return ne.evaluate('cosh(x)')
    
def logcosh(x):
    return -LOG2 + numpy.logaddexp(-x, x)
    
def acosh(x):
    y = numpy.clip(x, a_min=EPSILON+1)
    return ne.evaluate('arccosh(y)')
    
def logit(x):
    y = numpy.clip(x, a_min=EPSILON, a_max=1-EPSILON)
    return ne.evaluate('log(y/(1-y))')
    
def softplus(x):
    return numpy.logaddexp(0, x)
