optimizers
## class StochasticGradientDescent
StochasticGradientDescent
Basic algorithm of gradient descent with minibatches.
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, scheduler=<paysage.optimizers.PowerLawDecay object at 0x11ad0e898>, tolerance=0.001)

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
None
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





## class PowerLawDecay
None
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
None
### \_\_init\_\_
```py

def __init__(self, scheduler=<paysage.optimizers.PowerLawDecay object at 0x11ad0e828>, tolerance=0.001)

```



Initialize self.  See help(type(self)) for accurate signature.


### check\_convergence
```py

def check_convergence(self)

```





## class Scheduler
None
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
Momentum
Stochastic gradient descent with momentum.
Qian, N. (1999).
On the momentum term in gradient descent learning algorithms.
Neural Networks, 12(1), 145–151
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, momentum=0.9, scheduler=<paysage.optimizers.PowerLawDecay object at 0x11ad0e940>, tolerance=1e-06)

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
RMSProp
Geoffrey Hinton's Coursera Course Lecture 6e
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, mean_square_weight=0.9, scheduler=<paysage.optimizers.PowerLawDecay object at 0x11ad0e9e8>, tolerance=1e-06)

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
ADAM
Adaptive Moment Estimation algorithm.
Kingma, D. P., & Ba, J. L. (2015).
Adam: a Method for Stochastic Optimization.
International Conference on Learning Representations, 1–13.
### \_\_init\_\_
```py

def __init__(self, model, stepsize=0.001, mean_weight=0.9, mean_square_weight=0.999, scheduler=<paysage.optimizers.PowerLawDecay object at 0x11ad0ea90>, tolerance=1e-06)

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

### gradient
```py

def gradient(model, minibatch, samples)

```



### gradient\_magnitude
```py

def gradient_magnitude(grad)

```


