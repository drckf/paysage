# Documentation for Gradient_Util (gradient_util.py)

## class Gradient
Gradient(layers, weights)


## class partial
partial(func, *args, **keywords) - new function with partial application<br />of the given arguments and keywords.


## functions

### grad\_accumulate
```py

def grad_accumulate(func, grad)

```



Apply a function entrywise over a Gradient object,<br />accumulating the result.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable): function with one argument<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float


### grad\_apply
```py

def grad_apply(func, grad)

```



Apply a function entrywise over a Gradient object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable)<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Gradient


### grad\_apply\_
```py

def grad_apply_(func_, grad)

```



Apply a function entrywise over a Gradient object.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies elements of grad in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func_ (callable, in place operation)<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### grad\_flatten
```py

def grad_flatten(grad)

```



Returns a flat vector of gradient parameters<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;(tensor): vectorized gradient


### grad\_mapzip
```py

def grad_mapzip(func, grad1, grad2)

```



Apply a function entrywise over the zip of two Gradient objects.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func_ (callable, in place operation)<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Gradient


### grad\_mapzip\_
```py

def grad_mapzip_(func_, grad1, grad2)

```



Apply an in place function entrywise over the zip of two Gradient objects.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies elements of grad1 in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func_ (callable, in place operation)<br />&nbsp;&nbsp;&nbsp;&nbsp;grad1 (Gradient)<br />&nbsp;&nbsp;&nbsp;&nbsp;grad2 (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### grad\_norm
```py

def grad_norm(grad)

```



Compute the l2 norm of the gradient.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;magnitude (float)


### grad\_normalize\_
```py

def grad_normalize_(grad)

```



Normalize the gradient vector with respect to the L2 norm<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />Return:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### grad\_rms
```py

def grad_rms(grad)

```



Compute the root-mean-square of the gradient.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;rms (float)


### null\_grad
```py

def null_grad(model)

```



Return a gradient object filled with empty lists.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Gradient


### random\_grad
```py

def random_grad(model)

```



Return a gradient object filled with random numbers.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Gradient


### zero\_grad
```py

def zero_grad(model)

```



Return a gradient object filled with zero tensors.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a BoltzmannMachine object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Gradient

