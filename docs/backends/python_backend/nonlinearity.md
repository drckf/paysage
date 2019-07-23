# Documentation for Nonlinearity (nonlinearity.py)

## functions

### acosh
```py

def acosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (greater than 1): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise inverse hyperbolic cosine.


### atanh
```py

def atanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse hyperbolic tangent of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (between -1 and +1): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise inverse hyperbolic tangent


### cos
```py

def cos(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise cosine.


### cosh
```py

def cosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise hyperbolic cosine.


### exp
```py

def exp(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise exponential function of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (non-negative): Elementwise exponential.


### expit
```py

def expit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise expit (a.k.a. logistic) function of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise expit (a.k.a. logistic).


### log
```py

def log(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise natural logarithm of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (non-negative): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise natural logarithm.


### logaddexp
```py

def logaddexp(x1: numpy.ndarray, x2: numpy.ndarray) -> numpy.ndarray

```



Elementwise logaddexp function: log(exp(x1) + exp(x2))<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x1: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;x2: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise logaddexp.


### logcosh
```py

def logcosh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logarithm of the hyperbolic cosine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise logarithm of the hyperbolic cosine.


### logit
```py

def logit(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise logit function of a tensor. Inverse of the expit function.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (between 0 and 1): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise logit function


### normal\_pdf
```py

def normal_pdf(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise probability density function of the standard normal distribution.<br /><br />For the PDF of a normal distributon with mean u and standard deviation sigma, use<br />normal_pdf((x-u)/sigma) / sigma.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (tensor)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise pdf


### reciprocal
```py

def reciprocal(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise inverse of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (non-zero): A tensor:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise inverse.


### sin
```py

def sin(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise sine of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise sine.


### softmax
```py

def softmax(x: numpy.ndarray, axis: int = 1) -> numpy.ndarray

```



Softmax function on a tensor.<br />Exponentiaties the tensor elementwise and divides<br />&nbsp;&nbsp;&nbsp;&nbsp;by the sum along axis=1.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Softmax of the tensor.


### softplus
```py

def softplus(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise softplus function of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise softplus.


### sqrt
```py

def sqrt(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square root of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x (non-negative): A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor(non-negative): Elementwise square root.


### square
```py

def square(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise square of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (non-negative): Elementwise square.


### tabs
```py

def tabs(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise absolute value of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (non-negative): Absolute value of x.


### tanh
```py

def tanh(x: numpy.ndarray) -> numpy.ndarray

```



Elementwise hyperbolic tangent of a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise hyperbolic tangent.


### tmul
```py

def tmul(a: Union[int, float], x: numpy.ndarray) -> numpy.ndarray

```



Elementwise multiplication of tensor x by scalar a.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a: scalar.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise a * x.


### tmul\_
```py

def tmul_(a: Union[int, float], x: numpy.ndarray)

```



Elementwise multiplication of tensor x by scalar a.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifes x in place<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a: scalar.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### tpow
```py

def tpow(x: numpy.ndarray, a: float) -> numpy.ndarray

```



Elementwise power of a tensor x to power a.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;x: A tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;a: Power.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Elementwise x to the power of a.

