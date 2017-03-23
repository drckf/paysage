# Documentation for Fit (fit.py)

## class PersistentContrastiveDivergence
PersistentContrastiveDivergence<br />PCD-k algorithm for approximate maximum likelihood inference.<br /><br />Tieleman, Tijmen.<br />"Training restricted Boltzmann machines using approximations to the<br />likelihood gradient."<br />Proceedings of the 25th international conference on Machine learning.<br />ACM, 2008.
### \_\_init\_\_
```py

def __init__(self, model, abatch, optimizer, sampler, epochs, mcsteps=1, skip=100, metrics=['ReconstructionError', 'EnergyDistance'])

```



Initialize self.  See help(type(self)) for accurate signature.


### train
```py

def train(self)

```





## class ContrastiveDivergence
ContrastiveDivergence<br />CD-k algorithm for approximate maximum likelihood inference.<br /><br />Hinton, Geoffrey E.<br />"Training products of experts by minimizing contrastive divergence."<br />Neural computation 14.8 (2002): 1771-1800.<br /><br />Carreira-Perpinan, Miguel A., and Geoffrey Hinton.<br />"On Contrastive Divergence Learning."<br />AISTATS. Vol. 10. 2005.
### \_\_init\_\_
```py

def __init__(self, model, abatch, optimizer, sampler, epochs, mcsteps=1, skip=100, metrics=['ReconstructionError', 'EnergyDistance'])

```



Initialize self.  See help(type(self)) for accurate signature.


### train
```py

def train(self)

```





## class DrivenSequentialMC
A base class for the sequential Monte Carlo samplers
### \_\_init\_\_
```py

def __init__(self, amodel, beta_momentum=0.9, beta_scale=0.2, method='stochastic')

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_state
```py

def get_state(self)

```



### randomize\_state
```py

def randomize_state(self, shape)

```



Set up the inital states for each of the Markov Chains.<br />The initial state is randomly initalized.


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.


### update\_beta
```py

def update_beta(self)

```



update beta with an AR(1) process<br /><br /> ~  ~ 


### update\_state
```py

def update_state(self, steps)

```





## class ProgressMonitor
### \_\_init\_\_
```py

def __init__(self, skip, batch, metrics=['ReconstructionError', 'EnergyDistance'])

```



Initialize self.  See help(type(self)) for accurate signature.


### check\_progress
```py

def check_progress(self, model, t, store=False, show=False)

```





## class TrainingMethod
### \_\_init\_\_
```py

def __init__(self, model, abatch, optimizer, sampler, epochs, skip=100, metrics=['ReconstructionError', 'EnergyDistance'])

```



Initialize self.  See help(type(self)) for accurate signature.




## class SequentialMC
SequentialMC<br />Simple class for a sequential Monte Carlo sampler.
### \_\_init\_\_
```py

def __init__(self, amodel, method='stochastic')

```



Initialize self.  See help(type(self)) for accurate signature.


### get\_state
```py

def get_state(self)

```



### randomize\_state
```py

def randomize_state(self, shape)

```



Set up the inital states for each of the Markov Chains.<br />The initial state is randomly initalized.


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.


### update\_state
```py

def update_state(self, steps)

```





## class Sampler
A base class for the sequential Monte Carlo samplers
### \_\_init\_\_
```py

def __init__(self, amodel, method='stochastic', **kwargs)

```



Initialize self.  See help(type(self)) for accurate signature.


### randomize\_state
```py

def randomize_state(self, shape)

```



Set up the inital states for each of the Markov Chains.<br />The initial state is randomly initalized.


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.



