# Documentation for Optimizers (optimizers.py)

## class GradientMemory
Many optimizers like RMSProp or ADAM keep track of moving averages<br />of the gradients. This class computes the first two moments of the<br />gradients as running averages.
### \_\_init\_\_
```py

def __init__(self, mean_weight=0.9, mean_square_weight=0.0)

```



Create a gradient memory object to keep track of the first two<br />moments of the gradient.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_weight (float \in (0,1); optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;how strongly to weight the previous gradient<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_square_weight (float \in (0,1); optional)<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;how strongly to weight the square of the previous gradient<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;GradientMemory


### normalize
```py

def normalize(self, grad, unbiased=False)

```



Divide grad by the square root of the mean square gradient.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;A running average is biased due to autoregressive correlations<br />&nbsp;&nbsp;&nbsp;&nbsp;between adjacent timepoints. The bias can be corrected by<br />&nbsp;&nbsp;&nbsp;&nbsp;dividing the results by appropriate weights that reflect<br />&nbsp;&nbsp;&nbsp;&nbsp;the degree of autocorrelation.<br /><br />&nbsp;&nbsp;&nbsp;&nbsp;Acts like the identity function if mean_square_weight = 0.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (a Gradient object)<br />&nbsp;&nbsp;&nbsp;&nbsp;unbiased (bool): whether to unbias the estimates<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;normalized Gradient object


### reset
```py

def reset(self)

```



Reset the accululated mean and mean square gradients.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies mean_gradient and mean_square_gradient in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, grad)

```



Update the running average of the model gradients and the running<br />average of the squared model gradients.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies mean_weight and mean_square_weight attributes in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (a Gradient object)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_mean
```py

def update_mean(self, grad)

```



Update the running average of the model gradients.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (a Gradient object)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_mean\_square
```py

def update_mean_square(self, grad)

```



Update the running average of the squared model gradients.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (a Gradient object)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class Optimizer
Base class for the optimizer methods.
### \_\_init\_\_
```py

def __init__(self, stepsize=<paysage.schedules.Constant object>, tolerance=1e-07)

```



Create an optimizer object:<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;stepsize (generator; optional): the stepsize schedule<br />&nbsp;&nbsp;&nbsp;&nbsp;tolerance (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the gradient magnitude to declar convergence<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Optimizer


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: True if converged, False if not


### update\_lr
```py

def update_lr(self)

```



Update the current value of the stepsize:<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies stepsize attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class Gradient
Vanilla gradient optimizer
### \_\_init\_\_
```py

def __init__(self, stepsize=<paysage.schedules.Constant object>, tolerance=1e-07)

```



Create a gradient descent optimizer.<br /><br />Aliases:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;stepsize (generator; optional): the stepsize schedule<br />&nbsp;&nbsp;&nbsp;&nbsp;tolerance (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the gradient magnitude to declar convergence<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;StochasticGradientDescent


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: True if converged, False if not


### reset
```py

def reset(self)

```



Reset the gradient memory (does nothing for vanilla gradient).<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies gradient memory in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, model, grad)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Changes parameters of model in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;grad: a Gradient object<br />&nbsp;&nbsp;&nbsp;&nbsp;epoch (int): the current epoch<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_lr
```py

def update_lr(self)

```



Update the current value of the stepsize:<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies stepsize attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class Momentum
Stochastic gradient descent with momentum.<br />Qian, N. (1999).<br />On the momentum term in gradient descent learning algorithms.<br />Neural Networks, 12(1), 145–151
### \_\_init\_\_
```py

def __init__(self, stepsize=<paysage.schedules.Constant object>, momentum=0.9, tolerance=1e-07)

```



Create a stochastic gradient descent with momentum optimizer.<br /><br />Aliases:<br />&nbsp;&nbsp;&nbsp;&nbsp;momentum<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;stepsize (generator; optional): the stepsize schedule<br />&nbsp;&nbsp;&nbsp;&nbsp;momentum (float; optional): the amount of momentum<br />&nbsp;&nbsp;&nbsp;&nbsp;tolerance (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the gradient magnitude to declar convergence<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Momentum


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: True if converged, False if not


### reset
```py

def reset(self)

```



Reset the gradient memory.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies gradient memory in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, model, grad)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Changes parameters of model in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;grad: a Gradient object<br />&nbsp;&nbsp;&nbsp;&nbsp;epoch (int): the current epoch<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_lr
```py

def update_lr(self)

```



Update the current value of the stepsize:<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies stepsize attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class RMSProp
Stochastic gradient descent with RMSProp.<br />Geoffrey Hinton's Coursera Course Lecture 6e
### \_\_init\_\_
```py

def __init__(self, stepsize=<paysage.schedules.Constant object>, mean_square_weight=0.9, tolerance=1e-07)

```



Create a stochastic gradient descent with RMSProp optimizer.<br /><br />Aliases:<br />&nbsp;&nbsp;&nbsp;&nbsp;rmsprop<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;stepsize (generator; optional): the stepsize schedule<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_square_weight (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for computing the running average of the mean-square gradient<br />&nbsp;&nbsp;&nbsp;&nbsp;tolerance (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the gradient magnitude to declar convergence<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;RMSProp


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: True if converged, False if not


### reset
```py

def reset(self)

```



Reset the gradient memory.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies gradient memory in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, model, grad)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Changes parameters of model in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;grad: a Gradient object<br />&nbsp;&nbsp;&nbsp;&nbsp;epoch (int): the current epoch<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_lr
```py

def update_lr(self)

```



Update the current value of the stepsize:<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies stepsize attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class partial
partial(func, *args, **keywords) - new function with partial application<br />of the given arguments and keywords.


## class ADAM
Stochastic gradient descent with Adaptive Moment Estimation algorithm.<br /><br />Kingma, D. P., & Ba, J. L. (2015).<br />Adam: a Method for Stochastic Optimization.<br />International Conference on Learning Representations, 1–13.
### \_\_init\_\_
```py

def __init__(self, stepsize=<paysage.schedules.Constant object>, mean_weight=0.9, mean_square_weight=0.999, tolerance=1e-07)

```



Create a stochastic gradient descent with ADAM optimizer.<br /><br />Aliases:<br />&nbsp;&nbsp;&nbsp;&nbsp;adam<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;stepsize (generator; optional): the stepsize schedule<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_weight (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for computing the running average of the mean gradient<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_square_weight (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for computing the running average of the mean-square gradient<br />&nbsp;&nbsp;&nbsp;&nbsp;tolerance (float; optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the gradient magnitude to declar convergence<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;ADAM


### check\_convergence
```py

def check_convergence(self)

```



Check the convergence criterion.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;bool: True if converged, False if not


### reset
```py

def reset(self)

```



Reset the gradient memory.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies gradient memory in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, model, grad)

```



Update the model parameters with a gradient step.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Changes parameters of model in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object to optimize<br />&nbsp;&nbsp;&nbsp;&nbsp;grad: a Gradient object<br />&nbsp;&nbsp;&nbsp;&nbsp;epoch (int): the current epoch<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_lr
```py

def update_lr(self)

```



Update the current value of the stepsize:<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies stepsize attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None



