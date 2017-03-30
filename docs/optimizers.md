# Documentation for Optimizers (optimizers.py)

## class StochasticGradientDescent
StochasticGradientDescent<br />Basic algorithm of gradient descent with minibatches.
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, scheduler=<paysage.optimizers.PowerLawDecay object at 0x116d50a90>, tolerance=0.001)

```



Initialize self.  See help(type(self)) for accurate signature.


### check\_convergence
```py

def check_convergence(self)

```



### update
```py

def update(self, model, v_data, v_model, epoch)

```





## class ExponentialDecay
### \_\_init\_\_
```py

def __init__(self, lr_decay=0.9)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_lr
```py

def get_lr(self)

```



### increment
```py

def increment(self, epoch)

```





## class GradientMemory
Many optimizers like RMSProp or ADAM keep track of moving averages<br />of the gradients. This class computes the first two moments of the<br />gradients as running averages.
### \_\_init\_\_
```py

def __init__(self, mean_weight=0.9, mean_square_weight=0.0)

```



Initialize self.  See help(type(self)) for accurate signature.


### normalize
```py

def normalize(self, grad, unbiased=False)

```



Divide grad by the square root of the mean square gradient.


### update
```py

def update(self, grad)

```



Update the running average of the model gradients and the running<br />average of the squared model gradients.


### update\_mean
```py

def update_mean(self, grad)

```



Update the running average of the model gradients.


### update\_mean\_square
```py

def update_mean_square(self, grad)

```



Update the running average of the squared model gradients.




## class PowerLawDecay
### \_\_init\_\_
```py

def __init__(self, lr_decay=0.1)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_lr
```py

def get_lr(self)

```



### increment
```py

def increment(self, epoch)

```





## class Optimizer
### \_\_init\_\_
```py

def __init__(self, scheduler=<paysage.optimizers.PowerLawDecay object at 0x116d50a20>, tolerance=0.001)

```



Initialize self.  See help(type(self)) for accurate signature.


### check\_convergence
```py

def check_convergence(self)

```





## class Scheduler
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### increment
```py

def increment(self, epoch)

```





## class Momentum
Momentum<br />Stochastic gradient descent with momentum.<br />Qian, N. (1999).<br />On the momentum term in gradient descent learning algorithms.<br />Neural Networks, 12(1), 145–151
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, momentum=0.9, scheduler=<paysage.optimizers.PowerLawDecay object at 0x116d50b38>, tolerance=1e-06)

```



Initialize self.  See help(type(self)) for accurate signature.


### check\_convergence
```py

def check_convergence(self)

```



### update
```py

def update(self, model, v_data, v_model, epoch)

```





## class RMSProp
RMSProp<br />Geoffrey Hinton's Coursera Course Lecture 6e
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, mean_square_weight=0.9, scheduler=<paysage.optimizers.PowerLawDecay object at 0x116d50be0>, tolerance=1e-06)

```



Initialize self.  See help(type(self)) for accurate signature.


### check\_convergence
```py

def check_convergence(self)

```



### update
```py

def update(self, model, v_data, v_model, epoch)

```





## class ADAM
ADAM<br />Adaptive Moment Estimation algorithm.<br />Kingma, D. P., & Ba, J. L. (2015).<br />Adam: a Method for Stochastic Optimization.<br />International Conference on Learning Representations, 1–13.
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, mean_weight=0.9, mean_square_weight=0.999, scheduler=<paysage.optimizers.PowerLawDecay object at 0x116d50c88>, tolerance=1e-06)

```



Initialize self.  See help(type(self)) for accurate signature.


### check\_convergence
```py

def check_convergence(self)

```



### update
```py

def update(self, model, v_data, v_model, epoch)

```





## functions

### deepcopy
```py

def deepcopy(x, memo=None, _nil=[])

```



Deep copy operation on arbitrary Python objects.<br /><br />See the module's __doc__ string for more info.


### gradient\_magnitude
```py

def gradient_magnitude(grad) -> float

```



Compute the magnitude of the gradient.

