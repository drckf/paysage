# Documentation for Math_Utils (math_utils.py)


This module defines math utilities.


## class MeanVarianceArrayCalculator
An online numerically stable mean and variance calculator.<br />For calculations on arrays, where tensor objects are returned.<br />The variance over the 0-axis is computed.<br />Uses Welford's algorithm for the variance.<br />B.P. Welford, Technometrics 4(3):419–420.
### \_\_init\_\_
```py

def __init__(self)

```



Create MeanVarianceArrayCalculator object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ The MeanVarianceArrayCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br /> ~ Modifies the metrics in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, samples, axis=0) -> None

```



Update the online calculation of the mean and variance.<br /><br />Notes:<br /> ~ Modifies the metrics in place.<br /><br />Args:<br /> ~ samples: data samples<br /><br />Returns:<br /> ~ None




## class MeanVarianceCalculator
An online numerically stable mean and variance calculator.<br />For computations on vector objects, where single values are returned.<br />Uses Welford's algorithm for the variance.<br />B.P. Welford, Technometrics 4(3):419–420.
### \_\_init\_\_
```py

def __init__(self)

```



Create MeanVarianceCalculator object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ The MeanVarianceCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br /> ~ Modifies the metrics in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, samples) -> None

```



Update the online calculation of the mean and variance.<br /><br />Notes:<br /> ~ Modifies the metrics in place.<br /><br />Args:<br /> ~ samples: data samples<br /><br />Returns:<br /> ~ None




## class MeanArrayCalculator
An online mean calculator.<br />Calculates the mean of a tensor along axes.<br />Returns a tensor.
### \_\_init\_\_
```py

def __init__(self)

```



Create a MeanArrayCalculator object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ The MeanArrayCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br /> ~ Modifies the metric in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, samples, axis=0) -> None

```



Update the online calculation of the mean.<br /><br />Notes:<br /> ~ Modifies the metrics in place.<br /><br />Args:<br /> ~ samples: data samples<br /><br />Returns:<br /> ~ None




## class MeanCalculator
An online mean calculator.<br />Calculates the mean of tensors, returning a single number.
### \_\_init\_\_
```py

def __init__(self)

```



Create a MeanCalculator object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ The MeanCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br /> ~ Modifies the metric in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, samples) -> None

```



Update the online calculation of the mean.<br /><br />Notes:<br /> ~ Modifies the metrics in place.<br /><br />Args:<br /> ~ samples: data samples<br /><br />Returns:<br /> ~ None



