fit
## class PersistentContrastiveDivergence
PersistentContrastiveDivergence
PCD-k algorithm for approximate maximum likelihood inference.

Tieleman, Tijmen.
"Training restricted Boltzmann machines using approximations to the
likelihood gradient."
Proceedings of the 25th international conference on Machine learning.
ACM, 2008.
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
ContrastiveDivergence
CD-k algorithm for approximate maximum likelihood inference.

Hinton, Geoffrey E.
"Training products of experts by minimizing contrastive divergence."
Neural computation 14.8 (2002): 1771-1800.

Carreira-Perpinan, Miguel A., and Geoffrey Hinton.
"On Contrastive Divergence Learning."
AISTATS. Vol. 10. 2005.
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



Set up the inital states for each of the Markov Chains.
The initial state is randomly initalized.


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.


### update\_beta
```py

def update_beta(self)

```



update beta with an AR(1) process

        


### update\_state
```py

def update_state(self, steps)

```





## class ProgressMonitor
None
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
None
### \_\_init\_\_
```py

def __init__(self, model, abatch, optimizer, sampler, epochs, skip=100, metrics=['ReconstructionError', 'EnergyDistance'])

```



Initialize self.  See help(type(self)) for accurate signature.




## class SequentialMC
SequentialMC
Simple class for a sequential Monte Carlo sampler.
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



Set up the inital states for each of the Markov Chains.
The initial state is randomly initalized.


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



Set up the inital states for each of the Markov Chains.
The initial state is randomly initalized.


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.



