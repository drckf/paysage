# Documentation for Metrics (metrics.py)

## class ReconstructionError
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### reset
```py

def reset(self)

```



### update
```py

def update(self, minibatch=None, reconstructions=None, **kwargs)

```



### value
```py

def value(self)

```





## class EnergyDistance
### \_\_init\_\_
```py

def __init__(self, downsample=100)

```



Initialize self.  See help(type(self)) for accurate signature.


### reset
```py

def reset(self)

```



### update
```py

def update(self, minibatch=None, samples=None, **kwargs)

```



### value
```py

def value(self)

```





## class EnergyZscore
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### reset
```py

def reset(self)

```



### update
```py

def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs)

```



### value
```py

def value(self)

```





## class EnergyGap
### \_\_init\_\_
```py

def __init__(self)

```



Initialize self.  See help(type(self)) for accurate signature.


### reset
```py

def reset(self)

```



### update
```py

def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs)

```



### value
```py

def value(self)

```




