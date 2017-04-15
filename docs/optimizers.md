# Documentation for Optimizers (optimizers.py)

## class ExponentialDecay
Learning rate that decays exponentially per epoch
### \_\_init\_\_
```py

def __init__(self, lr_decay=0.9)

```



Create an exponential decay learning rate schedule.<br />Larger lr_decay -> slower decay.<br /><br />Args:<br /> ~ lr_decay (float \in (0,1))<br /><br />Returns:<br /> ~ ExponentialDecay


### get\_lr
```py

def get_lr(self)

```



Compute the current value of the learning rate.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ lr (float)


### increment
```py

def increment(self, epoch)

```



Update the iter and epoch attributes.<br /><br />Notes:<br /> ~ Modifies iter and epoch attributes in place.<br /><br />Args:<br /> ~ epoch (int): the current epoch<br /><br />Returns:<br /> ~ None




## class GradientMemory
Many optimizers like RMSProp or ADAM keep track of moving averages<br />of the gradients. This class computes the first two moments of the<br />gradients as running averages.
### \_\_init\_\_
```py

def __init__(self, mean_weight=0.9, mean_square_weight=0.0)

```



Create a gradient memory object to keep track of the first two<br />moments of the gradient.<br /><br />Args:<br /> ~ mean_weight (float \in (0,1); optional):<br /> ~  ~ how strongly to weight the previous gradient<br /> ~ mean_square_weight (float \in (0,1); optional)<br /> ~  ~ how strongly to weight the square of the previous gradient<br /><br />Returns:<br /> ~ GradientMemory


### normalize
```py

def normalize(self, grad, unbiased=False)

```



Divide grad by the square root of the mean square gradient.<br /><br />Notes:<br /> ~ A running average is biased due to autoregressive correlations<br /> ~ between adjacent timepoints. The bias can be corrected by<br /> ~ dividing the results by appropriate weights that reflect<br /> ~ the degree of autocorrelation.<br /><br /> ~ Acts like the identity function if mean_square_weight = 0.<br /><br />Args:<br /> ~ grad (a Gradient object)<br /> ~ unbiased (bool): whether to unbias the estimates<br /><br />Returns:<br /> ~ normalized Gradient object


### update
```py

def update(self, grad)

```



Update the running average of the model gradients and the running<br />average of the squared model gradients.<br /><br />Notes:<br /> ~ Modifies mean_weight and mean_square_weight attributes in place.<br /><br />Args:<br /> ~ grad (a Gradient object)<br /><br />Returns:<br /> ~ None


### update\_mean
```py

def update_mean(self, grad)

```



Update the running average of the model gradients.<br /><br />Args:<br /> ~ grad (a Gradient object)<br /><br />Returns:<br /> ~ None


### update\_mean\_square
```py

def update_mean_square(self, grad)

```



Update the running average of the squared model gradients.<br /><br />Args:<br /> ~ grad (a Gradient object)<br /><br />Returns:<br /> ~ None




## class PowerLawDecay
Learning rate that decays with a power law per epoch
### \_\_init\_\_
```py

def __init__(self, lr_decay=0.1)

```



Create a power law decay learning rate schedule.<br />Larger lr_decay -> faster decay.<br /><br />Args:<br /> ~ lr_decay (float \in (0,1))<br /><br />Returns:<br /> ~ PowerLawDecay


### get\_lr
```py

def get_lr(self)

```



Compute the current value of the learning rate.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ lr (float)


### increment
```py

def increment(self, epoch)

```



Update the iter and epoch attributes.<br /><br />Notes:<br /> ~ Modifies iter and epoch attributes in place.<br /><br />Args:<br /> ~ epoch (int): the current epoch<br /><br />Returns:<br /> ~ None




## class Optimizer
Base class for the optimizer methods.
### \_\_init\_\_
```py

def __init__(self, scheduler=<paysage.optimizers.PowerLawDecay object at 0x11ff48860>, tolerance=1e-07)

```



Create an optimizer object:<br /><br />Args:<br /> ~ scheduler (a learning rate schedule object; optional)<br /> ~ tolerance (float; optional):<br /> ~  ~ the gradient magnitude to declar convergence<br /><br />Returns:<br /> ~ Optimizer


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ bool: True if converged, False if not




## class Scheduler
Base class for the learning rate schedulers
### \_\_init\_\_
```py

def __init__(self)

```



Create a scheduler object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ Scheduler


### increment
```py

def increment(self, epoch)

```



Update the iter and epoch attributes.<br /><br />Notes:<br /> ~ Modifies iter and epoch attributes in place.<br /><br />Args:<br /> ~ epoch (int): the current epoch<br /><br />Returns:<br /> ~ None




## class Gradient
Vanilla gradient optimizer
### \_\_init\_\_
```py

def __init__(self, stepsize=0.001, scheduler=<paysage.optimizers.PowerLawDecay object at 0x127ac2048>, tolerance=1e-07, ascent=False)

```



Create a gradient ascent/descent optimizer.<br /><br />Aliases:<br /> ~ gradient<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ stepsize (float; optional): the initial stepsize<br /> ~ scheduler (a learning rate scheduler object; optional)<br /> ~ tolerance (float; optional):<br /> ~  ~ the gradient magnitude to declar convergence<br /><br />Returns:<br /> ~ StochasticGradientDescent


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ bool: True if converged, False if not


### update
```py

def update(self, model, grad, epoch)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br /> ~ Changes parameters of model in place.<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ grad: a Gradient object<br /> ~ epoch (int): the current epoch<br /><br />Returns:<br /> ~ None




## class Momentum
Stochastic gradient descent with momentum.<br />Qian, N. (1999).<br />On the momentum term in gradient descent learning algorithms.<br />Neural Networks, 12(1), 145–151
### \_\_init\_\_
```py

def __init__(self, stepsize=0.001, momentum=0.9, scheduler=<paysage.optimizers.PowerLawDecay object at 0x127ac2080>, tolerance=1e-07, ascent=False)

```



Create a stochastic gradient descent with momentum optimizer.<br /><br />Aliases:<br /> ~ momentum<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ stepsize (float; optional): the initial stepsize<br /> ~ momentum (float; optional): the amount of momentum<br /> ~ scheduler (a learning rate scheduler object; optional)<br /> ~ tolerance (float; optional):<br /> ~  ~ the gradient magnitude to declar convergence<br /><br />Returns:<br /> ~ Momentum


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ bool: True if converged, False if not


### update
```py

def update(self, model, grad, epoch)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br /> ~ Changes parameters of model in place.<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ grad: a Gradient object<br /> ~ epoch (int): the current epoch<br /><br />Returns:<br /> ~ None




## class RMSProp
Stochastic gradient descent with RMSProp.<br />Geoffrey Hinton's Coursera Course Lecture 6e
### \_\_init\_\_
```py

def __init__(self, stepsize=0.001, mean_square_weight=0.9, scheduler=<paysage.optimizers.PowerLawDecay object at 0x127ac3668>, tolerance=1e-07, ascent=False)

```



Create a stochastic gradient descent with RMSProp optimizer.<br /><br />Aliases:<br /> ~ rmsprop<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ stepsize (float; optional): the initial stepsize<br /> ~ mean_square_weight (float; optional):<br /> ~  ~ for computing the running average of the mean-square gradient<br /> ~ scheduler (a learning rate scheduler object; optional)<br /> ~ tolerance (float; optional):<br /> ~  ~ the gradient magnitude to declar convergence<br /><br />Returns:<br /> ~ RMSProp


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ bool: True if converged, False if not


### update
```py

def update(self, model, grad, epoch)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br /> ~ Changes parameters of model in place.<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ grad: a Gradient object<br /> ~ epoch (int): the current epoch<br /><br />Returns:<br /> ~ None




## class partial
partial(func, *args, **keywords) - new function with partial application<br />of the given arguments and keywords.


## class ADAM
Stochastic gradient descent with Adaptive Moment Estimation algorithm.<br /><br />Kingma, D. P., & Ba, J. L. (2015).<br />Adam: a Method for Stochastic Optimization.<br />International Conference on Learning Representations, 1–13.
### \_\_init\_\_
```py

def __init__(self, stepsize=0.001, mean_weight=0.9, mean_square_weight=0.999, scheduler=<paysage.optimizers.PowerLawDecay object at 0x127ac3710>, tolerance=1e-07, ascent=False)

```



Create a stochastic gradient descent with ADAM optimizer.<br /><br />Aliases:<br /> ~ adam<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ stepsize (float; optional): the initial stepsize<br /> ~ mean_weight (float; optional):<br /> ~  ~ for computing the running average of the mean gradient<br /> ~ mean_square_weight (float; optional):<br /> ~  ~ for computing the running average of the mean-square gradient<br /> ~ scheduler (a learning rate scheduler object; optional)<br /> ~ tolerance (float; optional):<br /> ~  ~ the gradient magnitude to declar convergence<br /><br />Returns:<br /> ~ ADAM


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ bool: True if converged, False if not


### update
```py

def update(self, model, grad, epoch)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br /> ~ Changes parameters of model in place.<br /><br />Args:<br /> ~ model: a Model object to optimize<br /> ~ grad: a Gradient object<br /> ~ epoch (int): the current epoch<br /><br />Returns:<br /> ~ None



