# Documentation for Math_Utils (math_utils.py)


This module defines math utilities.


## class MeanVarianceCalculator
An online numerically stable mean and variance calculator.<br />Uses Welford's algorithm for the variance.<br />B.P. Welford, Technometrics 4(3):419–420.
### \_\_init\_\_
```py

def __init__(self)

```



Create MeanVarianceCalculator object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ The MeanVarianceCalculator object.


### calculate
```py

def calculate(self, samples)

```



Run an online calculation of the mean and variance.<br /><br />Notes:<br /> ~ The unnormalized variance is calculated<br /> ~  ~ (not divided by the number of samples).<br /><br />Args:<br /> ~ samples: data samples<br /><br />Returns:<br /> ~ None


### reset
```py

def reset(self) -> None

```



Resets the calculation to the initial state.<br /><br />Note:<br /> ~ Modifies the metric in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None



