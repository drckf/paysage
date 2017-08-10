# Documentation for Penalties (penalties.py)

## class logdet_penalty
Base penalty class.<br />Derived classes should define `value` and `grad` functions.
### \_\_init\_\_
```py

def __init__(self, penalty)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.




## class log_penalty
Base penalty class.<br />Derived classes should define `value` and `grad` functions.
### \_\_init\_\_
```py

def __init__(self, penalty)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.




## class l1_penalty
Base penalty class.<br />Derived classes should define `value` and `grad` functions.
### \_\_init\_\_
```py

def __init__(self, penalty)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.




## class l2_penalty
Base penalty class.<br />Derived classes should define `value` and `grad` functions.
### \_\_init\_\_
```py

def __init__(self, penalty)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.




## class Penalty
Base penalty class.<br />Derived classes should define `value` and `grad` functions.
### \_\_init\_\_
```py

def __init__(self, penalty)

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.




## functions

### from\_config
```py

def from_config(config)

```



Builds an instance from a config.

