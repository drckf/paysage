# Documentation for Metrics (metrics.py)


This module defines classes that represent the state of some model fit metric,
derived from summary information about the current state of the model
(encapsulated in MetricState).


## class ReconstructionError
Compute the root-mean-squared error between observations and their<br />reconstructions using minibatches.
### \_\_init\_\_
```py

def __init__(self)

```



Create a ReconstructionError object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ ReconstructionERror


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, update_args: paysage.metrics.MetricState) -> None

```



Update the estimate for the reconstruction error using a batch<br />of observations and a batch of reconstructions.<br /><br />Args:<br /> ~ update_args: uses visible layer of minibatch and reconstructions<br /><br />Returns:<br /> ~ None


### value
```py

def value(self) -> float

```



Get the value of the reconstruction error.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ reconstruction error (float)




## class EnergyDistance
Compute the energy distance between two distributions using<br />minibatches of sampled configurations.<br /><br />Szekely, G.J. (2002)<br />E-statistics: The Energy of Statistical Samples.<br />Technical Report BGSU No 02-16.
### \_\_init\_\_
```py

def __init__(self)

```



Create EnergyDistance object.<br /><br />Args:<br /> ~ downsample (int; optional): how many samples to use<br /><br />Returns:<br /> ~ energy distance object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, update_args: paysage.metrics.MetricState) -> None

```



Update the estimate for the energy distance using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br /> ~ update_args: uses visible layer of minibatch and samples<br /><br />Returns:<br /> ~ None


### value
```py

def value(self) -> float

```



Get the value of the energy distance.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy distance (float)




## class WeightSparsity
Compute the weight sparsity of the model as the formula<br /><br />p = \sum_j(\sum_i w_ij^2)^2/\sum_i w_ij^4<br /><br />Tubiana, J., Monasson, R. (2017)<br />Emergence of Compositional Representations in Restricted Boltzmann Machines,<br />PRL 118, 138301 (2017)
### \_\_init\_\_
```py

def __init__(self)

```



Create WeightSparsity object.<br /><br />Args:<br /> ~ None<br />Returns:<br /> ~ WeightSparsity object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, update_args: paysage.metrics.MetricState) -> None

```



Compute the weight sparsity of the model<br /><br />Args:<br /> ~ update_args: uses model only<br /><br />Returns:<br /> ~ None


### value
```py

def value(self) -> float

```



Get the value of the weight sparsity.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ weight sparsity (float)




## class EnergyZscore
Samples drawn from a model should have much lower energy<br />than purely random samples. The "energy gap" is the average<br />energy difference between samples from the model and random<br />samples. The "energy z-score" is the energy gap divided by<br />the standard deviation of the energy taken over random<br />samples.
### \_\_init\_\_
```py

def __init__(self)

```



Create an EnergyZscore object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ EnergyZscore object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, update_args: paysage.metrics.MetricState) -> None

```



Update the estimate for the energy z-score using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br /> ~ update_args: uses all layers of minibatch and random_samples, and model<br /><br />Returns:<br /> ~ None


### value
```py

def value(self) -> float

```



Get the value of the energy z-score.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy z-score (float)




## class HeatCapacity
Compute the heat capacity of the system thought of as a spin system.<br /><br />We take the HC to be the second cumulant of the energy, or alternately<br />the negative second derivative with respect to inverse temperature of<br />the Gibbs free energy.  In order to estimate this quantity we perform<br />Gibbs sampling starting from random samples drawn from the visible layer's<br />distribution.
### \_\_init\_\_
```py

def __init__(self)

```



Create HeatCapacity object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ The HeatCapacity object.


### update
```py

def update(self, update_args: paysage.metrics.MetricState) -> None

```



Update the estimate for the heat capacity.<br /><br />Args:<br /> ~ update_args: uses all layers of random_samples, and model<br /><br />Returns:<br /> ~ None


### value
```py

def value(self) -> float

```



Get the value of the heat capacity.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ heat capacity (float)




## class WeightSquare
Compute the mean squared weights of the model per hidden unit<br /><br />w2 = 1/(#hidden units)*\sum_ij w_ij^2<br /><br />Tubiana, J., Monasson, R. (2017)<br />Emergence of Compositional Representations in Restricted Boltzmann Machines,<br />PRL 118, 138301 (2017)
### \_\_init\_\_
```py

def __init__(self)

```



Create WeightSquare object.<br /><br />Args:<br /> ~ None<br />Returns:<br /> ~ WeightSquare object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, update_args: paysage.metrics.MetricState) -> None

```



Compute the weight square of the model<br /><br />Args:<br /> ~ update_args: uses model only<br /><br />Returns:<br /> ~ None


### value
```py

def value(self) -> float

```



Get the value of the weight sparsity.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ weight sparsity (float)




## class MetricState
MetricState(minibatch, reconstructions, random_samples, samples, model)


## class EnergyGap
Samples drawn from a model should have much lower energy<br />than purely random samples. The "energy gap" is the average<br />energy difference between samples from the model and random<br />samples.
### \_\_init\_\_
```py

def __init__(self)

```



Create an EnergyGap object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy gap object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, update_args: paysage.metrics.MetricState) -> None

```



Update the estimate for the energy gap using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br /> ~ update_args: uses all layers of minibatch and random_samples, and model<br /><br />Returns:<br /> ~ None


### value
```py

def value(self)

```



Get the value of the energy gap.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy gap (float)




## functions

### namedtuple
```py

def namedtuple(typename, field_names, *, verbose=False, rename=False, module=None)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)

