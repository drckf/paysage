# Documentation for Fit (fit.py)

## class StochasticGradientDescent
Stochastic gradient descent with minibatches
### \_\_init\_\_
```py

def __init__(self, model, batch, optimizer, epochs, method=<function peristent_contrastive_divergence at 0x127acd400>, sampler=<class 'paysage.fit.SequentialMC'>, mcsteps=1, monitor=None)

```



Create a StochasticGradientDescent object.<br /><br />Args:<br /> ~ model: a model object<br /> ~ batch: a batch object<br /> ~ optimizer: an optimizer object<br /> ~ epochs (int): the number of epochs<br /> ~ method (optional): the method used to approximate the likelihood<br /> ~  ~  ~  ~  ~    gradient [cd, pcd, ortap]<br /> ~ sampler (optional): a sampler object<br /> ~ mcsteps (int, optional): the number of Monte Carlo steps per gradient<br /> ~ monitor (optional): a progress monitor<br /><br />Returns:<br /> ~ StochasticGradientDescent


### train
```py

def train(self)

```



Train the model.<br /><br />Notes:<br /> ~ Updates the model parameters in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None




## class DrivenSequentialMC
An accelerated sequential Monte Carlo sampler
### \_\_init\_\_
```py

def __init__(self, model, beta_momentum=0.9, beta_std=0.2, method='stochastic')

```



Create a sequential Monte Carlo sampler.<br /><br />Args:<br /> ~ model: a model object<br /> ~ beta_momentum (float in [0,1]): autoregressive coefficient of beta<br /> ~ beta_std (float > 0): the standard deviation of beta<br /> ~ method (str; optional): how to update the particles<br /><br />Returns:<br /> ~ SequentialMC


### get\_state
```py

def get_state(self)

```



Return the state attribute.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ state (tensor)


### randomize\_state
```py

def randomize_state(self, shape)

```



Set up the inital states for each of the Markov Chains.<br />The initial state is randomly initalized.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ shape (tuple): shape if the visible layer<br /><br />Returns:<br /> ~ None


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.<br /><br />Notes:<br /> ~ Modifies state attribute in place.<br /><br />Args:<br /> ~ tensor: the observed visible units<br /><br />Returns:<br /> ~ None


### update\_state
```py

def update_state(self, steps)

```



Update the state of the particles.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /> ~ Calls _update_beta() method.<br /><br />Args:<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ None




## class ProgressMonitor
Monitor the progress of training by computing statistics on the<br />validation set.
### \_\_init\_\_
```py

def __init__(self, batch, metrics=['ReconstructionError'])

```



Create a progress monitor.<br /><br />Args:<br /> ~ batch (int): the<br /> ~ metrics (list[str]): list of metrics to compute<br /><br />Returns:<br /> ~ ProgressMonitor


### check\_progress
```py

def check_progress(self, model, store=False, show=False)

```



Compute the metrics from a model on the validaiton set.<br /><br />Args:<br /> ~ model: a model object<br /> ~ store (bool): if true, store the metrics in a list<br /> ~ show (bool): if true, print the metrics to the screen<br /><br />Returns:<br /> ~ metdict (dict): an ordered dictionary with the metrics




## class SequentialMC
Basic sequential Monte Carlo sampler
### \_\_init\_\_
```py

def __init__(self, model, method='stochastic')

```



Create a sequential Monte Carlo sampler.<br /><br />Args:<br /> ~ model: a model object<br /> ~ method (str; optional): how to update the particles<br /><br />Returns:<br /> ~ SequentialMC


### get\_state
```py

def get_state(self)

```



Return the state attribute.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ state (tensor)


### randomize\_state
```py

def randomize_state(self, shape)

```



Set up the inital states for each of the Markov Chains.<br />The initial state is randomly initalized.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ shape (tuple): shape if the visible layer<br /><br />Returns:<br /> ~ None


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.<br /><br />Notes:<br /> ~ Modifies state attribute in place.<br /><br />Args:<br /> ~ tensor: the observed visible units<br /><br />Returns:<br /> ~ None


### update\_state
```py

def update_state(self, steps)

```



Update the state of the particles.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ None




## class OrderedDict
Dictionary that remembers insertion order


## class Sampler
Base class for the sequential Monte Carlo samplers
### \_\_init\_\_
```py

def __init__(self, model, method='stochastic', **kwargs)

```



Create a sampler.<br /><br />Args:<br /> ~ model: a model object<br /> ~ method (str; optional): how to update the particles<br /> ~ kwargs (optional)<br /><br />Returns:<br /> ~ sampler


### randomize\_state
```py

def randomize_state(self, shape)

```



Set up the inital states for each of the Markov Chains.<br />The initial state is randomly initalized.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ shape (tuple): shape if the visible layer<br /><br />Returns:<br /> ~ None


### set\_state
```py

def set_state(self, tensor)

```



Set up the inital states for each of the Markov Chains.<br /><br />Notes:<br /> ~ Modifies state attribute in place.<br /><br />Args:<br /> ~ tensor: the observed visible units<br /><br />Returns:<br /> ~ None




## functions

### cd
```py

def cd(vdata, model, sampler, steps=1)

```



Compute an approximation to the likelihood gradient using the CD-k<br />algorithm for approximate maximum likelihood inference.<br /><br />Hinton, Geoffrey E.<br />"Training products of experts by minimizing contrastive divergence."<br />Neural computation 14.8 (2002): 1771-1800.<br /><br />Carreira-Perpinan, Miguel A., and Geoffrey Hinton.<br />"On Contrastive Divergence Learning."<br />AISTATS. Vol. 10. 2005.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ sampler: a sampler object<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### contrastive\_divergence
```py

def contrastive_divergence(vdata, model, sampler, steps=1)

```



Compute an approximation to the likelihood gradient using the CD-k<br />algorithm for approximate maximum likelihood inference.<br /><br />Hinton, Geoffrey E.<br />"Training products of experts by minimizing contrastive divergence."<br />Neural computation 14.8 (2002): 1771-1800.<br /><br />Carreira-Perpinan, Miguel A., and Geoffrey Hinton.<br />"On Contrastive Divergence Learning."<br />AISTATS. Vol. 10. 2005.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ sampler: a sampler object<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### pcd
```py

def pcd(vdata, model, sampler, steps=1)

```



PCD-k algorithm for approximate maximum likelihood inference.<br /><br />Tieleman, Tijmen.<br />"Training restricted Boltzmann machines using approximations to the<br />likelihood gradient."<br />Proceedings of the 25th international conference on Machine learning.<br />ACM, 2008.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ sampler: a sampler object<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### peristent\_contrastive\_divergence
```py

def peristent_contrastive_divergence(vdata, model, sampler, steps=1)

```



PCD-k algorithm for approximate maximum likelihood inference.<br /><br />Tieleman, Tijmen.<br />"Training restricted Boltzmann machines using approximations to the<br />likelihood gradient."<br />Proceedings of the 25th international conference on Machine learning.<br />ACM, 2008.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ sampler: a sampler object<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### tap
```py

def tap(vdata, model, sampler=None, steps=None)

```



Compute the gradient using the Thouless-Anderson-Palmer (TAP)<br />mean field approximation.<br /><br />Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,<br />and Florent Krzakala<br />"A Deterministic and Generalized Framework for Unsupervised Learning<br />with Restricted Boltzmann Machines"<br /><br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ sampler (default to None): not required<br /> ~ steps (default to None): not requires<br /><br />Returns:<br /> ~ gradient

