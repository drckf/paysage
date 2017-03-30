# Documentation for Metrics (metrics.py)

## class ReconstructionError
Compute the root-mean-squared error between observations and their<br />reconstructions using minibatches.
### \_\_init\_\_
```py

def __init__(self)

```



Create a ReconstructionError object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ ReconstructionERror


### reset
```py

def reset(self)

```



Reset the metric to it's initial state.<br /><br />Notes:<br /> ~ Changes norm and mean_square_error in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, minibatch=None, reconstructions=None, **kwargs)

```



Update the estimate for the reconstruction error using a batch<br />of observations and a batch of reconstructions.<br /><br />Notes:<br /> ~ Changes norm and mean_square_error in place.<br /><br />Args:<br /> ~ minibatch (tensor (num_samples, num_units))<br /> ~ reconstructions (tensor (num_samples, num))<br /> ~ kwargs: key word arguments<br /> ~  ~ not used, but helpful for looping through metric functions<br /><br />Returns:<br /> ~ None


### value
```py

def value(self)

```



Get the value of the reconstruction error.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ reconstruction error (float)




## class EnergyDistance
Compute the energy distance between two distributions using<br />minibatches of sampled configurations.<br /><br />Szekely, G.J. (2002)<br />E-statistics: The Energy of Statistical Samples.<br />Technical Report BGSU No 02-16.
### \_\_init\_\_
```py

def __init__(self, downsample=100)

```



Create EnergyDistance object.<br /><br />Args:<br /> ~ downsample (int; optional): how many samples to use<br /><br />Returns:<br /> ~ energy distance object


### reset
```py

def reset(self)

```



Reset the metric to it's initial state.<br /><br />Note:<br /> ~ Modifies norm and energy_distance in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, minibatch=None, samples=None, **kwargs)

```



Update the estimate for the energy distance using a batch<br />of observations and a batch of fantasy particles.<br /><br />Notes:<br /> ~ Changes norm and energy_distance in place.<br /><br />Args:<br /> ~ minibatch (tensor (num_samples, num_units))<br /> ~ samples (tensor (num_samples, num)): fantasy particles<br /> ~ kwargs: key word arguments<br /> ~  ~ not used, but helpful for looping through metric functions<br /><br />Returns:<br /> ~ None


### value
```py

def value(self)

```



Get the value of the energy distance.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy distance (float)




## class EnergyZscore
Samples drawn from a model should have much lower energy<br />than purely random samples. The "energy gap" is the average<br />energy difference between samples from the model and random<br />samples. The "energy z-score" is the energy gap divided by<br />the standard deviation of the energy taken over random<br />samples.
### \_\_init\_\_
```py

def __init__(self)

```



Create an EnergyZscore object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy z-score object


### reset
```py

def reset(self)

```



Reset the metric to it's initial state.<br /><br />Note:<br /> ~ Modifies norm, random_mean, and random_mean_square in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs)

```



Update the estimate for the energy z-score using a batch<br />of observations and a batch of fantasy particles.<br /><br />Notes:<br /> ~ Changes norm, random_mean, and random_mean_square in place.<br /><br />Args:<br /> ~ minibatch (tensor (num_samples, num_units)):<br /> ~  ~ samples from the model<br /> ~ random_samples (tensor (num_samples, num))<br /> ~ amodel (Model): the model<br /> ~ kwargs: key word arguments<br /> ~  ~ not used, but helpful for looping through metric functions<br /><br />Returns:<br /> ~ None


### value
```py

def value(self)

```



Get the value of the energy z-score.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy z-score (float)




## class EnergyGap
Samples drawn from a model should have much lower energy<br />than purely random samples. The "energy gap" is the average<br />energy difference between samples from the model and random<br />samples.
### \_\_init\_\_
```py

def __init__(self)

```



Create an EnergyGap object.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy gap object


### reset
```py

def reset(self)

```



Reset the metric to it's initial state.<br /><br />Note:<br /> ~ Modifies norm and energy_gap in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, minibatch=None, random_samples=None, amodel=None, **kwargs)

```



Update the estimate for the energy gap using a batch<br />of observations and a batch of fantasy particles.<br /><br />Notes:<br /> ~ Changes norm and energy_gap in place.<br /><br />Args:<br /> ~ minibatch (tensor (num_samples, num_units)):<br /> ~  ~ samples from the model<br /> ~ random_samples (tensor (num_samples, num))<br /> ~ amodel (Model): the model<br /> ~ kwargs: key word arguments<br /> ~  ~ not used, but helpful for looping through metric functions<br /><br />Returns:<br /> ~ None


### value
```py

def value(self)

```



Get the value of the energy gap.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ energy gap (float)



