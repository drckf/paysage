# Documentation for Online_Moments (online_moments.py)


This module defines math utilities.


## class MeanVarianceArrayCalculator
An online numerically stable mean and variance calculator.<br />For calculations on arrays, where tensor objects are returned.<br />The variance over the 0-axis is computed.<br />Uses Welford's algorithm for the variance.<br />B.P. Welford, Technometrics 4(3):419–420.
### \_\_init\_\_
```py

def __init__(self)

```



Create MeanVarianceArrayCalculator object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;The MeanVarianceArrayCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metrics in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### to\_dataframe
```py

def to_dataframe(self)

```



Create a config DataFrame for the object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;df (DataFrame): a DataFrame representation of the object.


### update
```py

def update(self, samples, axis=0) -> None

```



Update the online calculation of the mean and variance.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metrics in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;samples: data samples<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class MeanVarianceCalculator
An online numerically stable mean and variance calculator.<br />For computations on vector objects, where single values are returned.<br />Uses Welford's algorithm for the variance.<br />B.P. Welford, Technometrics 4(3):419–420.
### \_\_init\_\_
```py

def __init__(self)

```



Create MeanVarianceCalculator object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;The MeanVarianceCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metrics in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, samples) -> None

```



Update the online calculation of the mean and variance.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metrics in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;samples: data samples<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class MeanArrayCalculator
An online mean calculator.<br />Calculates the mean of a tensor along axes.<br />Returns a tensor.
### \_\_init\_\_
```py

def __init__(self)

```



Create a MeanArrayCalculator object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;The MeanArrayCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metric in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, samples, axis=0) -> None

```



Update the online calculation of the mean.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metrics in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;samples: data samples<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class MeanCalculator
An online mean calculator.<br />Calculates the mean of tensors, returning a single number.
### \_\_init\_\_
```py

def __init__(self)

```



Create a MeanCalculator object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;The MeanCalculator object.


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metric in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, samples) -> None

```



Update the online calculation of the mean.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metrics in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;samples (tensor): data samples<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None



