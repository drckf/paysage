import torch

from . import matrix
from . import typedef as T

LOG2 = 0.6931471805599453
SQRT2PI = 2.5066282746310002

def tmul(a: T.Scalar, x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise multiplication of tensor x by scalar a.

    Args:
        x: A tensor.
        a: scalar.

    Returns:
        tensor: Elementwise a * x.

    """
    return x.mul(a)

def tmul_(a: T.Scalar, x: T.FloatTensor) -> T.FloatTensor:
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
    x.mul_(a)

def tabs(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise absolute value of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor (non-negative): Absolute value of x.

    """
    return torch.abs(x)

def exp(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise exponential function of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor (non-negative): Elementwise exponential.

    """
    return torch.exp(x)

def log(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise natural logarithm of a tensor.

    Args:
        x (non-negative): A tensor.

    Returns:
        tensor: Elementwise natural logarithm.

    """
    y = matrix.clip(x, a_min=T.EPSILON)
    return torch.log(y)

def tanh(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise hyperbolic tangent of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise hyperbolic tangent.

    """
    return torch.tanh(x)

def expit(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise expit (a.k.a. logistic) function of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise expit (a.k.a. logistic).

    """
    return 0.5 * (1.0 + tanh(0.5 * x))

def softmax(x: T.Tensor, axis: int = 1) -> T.Tensor:
    """
    Softmax function on a tensor.
    Exponentiaties the tensor elementwise and divides
        by the sum along axis.

    Args:
        x: A tensor.

    Returns:
        tensor: Softmax of the tensor.

    """
    xreg = matrix.subtract(matrix.tmax(x, axis=axis, keepdims=True), x)
    y = exp(xreg)
    return matrix.divide(matrix.tsum(y, axis=axis, keepdims=True), y)

def reciprocal(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise inverse of a tensor.

    Args:
        x (non-zero): A tensor:

    Returns:
        tensor: Elementwise inverse.

    """
    return torch.reciprocal(x)

def atanh(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise inverse hyperbolic tangent of a tensor.

    Args:
        x (between -1 and +1): A tensor.

    Returns:
        tensor: Elementwise inverse hyperbolic tangent

    """
    y = matrix.clip(x, a_min=T.EPSILON - 1, a_max=1 -T.EPSILON)
    return (log(1+y) - log(1-y)) / 2

def sqrt(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise square root of a tensor.

    Args:
        x (non-negative): A tensor.

    Returns:
        tensor(non-negative): Elementwise square root.

    """
    return torch.sqrt(x)

def square(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise square of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor (non-negative): Elementwise square.

    """
    return x * x

def tpow(x: T.FloatTensor, a: float) -> T.FloatTensor:
    """
    Elementwise power of a tensor x to power a.

    Args:
        x: A tensor.
        a: Power.

    Returns:
        tensor: Elementwise x to the power of a.

    """
    return torch.pow(x, a)

def cosh(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise hyperbolic cosine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise hyperbolic cosine.

    """
    return torch.cosh(x)

def logaddexp(x1: T.FloatTensor, x2: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise logaddexp function: log(exp(x1) + exp(x2))

    Args:
        x1: A tensor.
        x2: A tensor.

    Returns:
        tensor: Elementwise logaddexp.

    """
    # log(exp(x1) + exp(x2))
    # = log( exp(x1) (1 + exp(x2 - x1))) = x1 + log(1 + exp(x2 - x1))
    # = log( exp(x2) (exp(x1 - x2) + 1)) = x2 + log(1 + exp(x1 - x2))
    diff = torch.min(x2 - x1, x1 - x2)
    return torch.max(x1, x2) + torch.log1p(exp(diff))

def logcosh(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise logarithm of the hyperbolic cosine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise logarithm of the hyperbolic cosine.

    """
    return -LOG2 + logaddexp(-x, x)

def acosh(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise inverse hyperbolic cosine of a tensor.

    Args:
        x (greater than 1): A tensor.

    Returns:
        tensor: Elementwise inverse hyperbolic cosine.

    """
    y = matrix.clip(x, a_min=1+T.EPSILON)
    return log(y + sqrt(y+1) * sqrt(y-1))

def logit(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise logit function of a tensor. Inverse of the expit function.

    Args:
        x (between 0 and 1): A tensor.

    Returns:
        tensor: Elementwise logit function

    """
    y = matrix.clip(x, a_min=T.EPSILON, a_max = 1 - T.EPSILON)
    return torch.log(y / (1 - y))

def softplus(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise softplus function of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise softplus.

    """
    return logaddexp(matrix.zeros_like(x),x)

def cos(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise cosine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise cosine.

    """
    return torch.cos(x)

def sin(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise sine of a tensor.

    Args:
        x: A tensor.

    Returns:
        tensor: Elementwise sine.

    """
    return torch.sin(x)

def normal_pdf(x: T.FloatTensor) -> T.FloatTensor:
    """
    Elementwise probability density function of the standard normal distribution.

    For the PDF of a normal distributon with mean u and standard deviation sigma, use
    normal_pdf((x-u)/sigma) / sigma.

    Args:
        x (tensor)

    Returns:
        tensor: Elementwise pdf

    """
    return exp(-0.5*x**2)/SQRT2PI
