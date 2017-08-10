# Documentation for Fit (fit.py)

## class StochasticGradientDescent
Stochastic gradient descent with minibatches
### \_\_init\_\_
```py

def __init__(self, model, batch, optimizer, epochs, sampler, method=<function persistent_contrastive_divergence at 0x11e805950>, mcsteps=1, monitor=None)

```



Create a StochasticGradientDescent object.<br /><br />Args:<br /> ~ model: a model object<br /> ~ batch: a batch object<br /> ~ optimizer: an optimizer object<br /> ~ epochs (int): the number of epochs<br /> ~ sampler: a sampler object<br /> ~ method (optional): the method used to approximate the likelihood<br /> ~  ~  ~  ~  ~    gradient [cd, pcd, tap]<br /> ~ mcsteps (int, optional): the number of Monte Carlo steps per gradient<br /> ~ monitor (optional): a progress monitor<br /><br />Returns:<br /> ~ StochasticGradientDescent


### train
```py

def train(self)

```



Train the model.<br /><br />Notes:<br /> ~ Updates the model parameters in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None




## class DrivenSequentialMC
An accelerated sequential Monte Carlo sampler
### \_\_init\_\_
```py

def __init__(self, model, clamped=None, updater='markov_chain', beta_momentum=0.9, beta_std=0.6, schedule=<paysage.schedules.Constant object at 0x11e754208>)

```



Create a sequential Monte Carlo sampler.<br /><br />Args:<br /> ~ model: a model object<br /> ~ beta_momentum (float in [0,1]): autoregressive coefficient of beta<br /> ~ beta_std (float > 0): the standard deviation of beta<br /> ~ schedule (generator; optional)<br /><br />Returns:<br /> ~ DrivenSequentialMC


### reset
```py

def reset(self)

```



Reset the sampler state.<br /><br />Notes:<br /> ~ Modifies sampler.state attribute in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### set\_state
```py

def set_state(self, state)

```



Set the state.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ state (State): The state of the units.<br /><br />Returns:<br /> ~ None


### set\_state\_from\_batch
```py

def set_state_from_batch(self, batch)

```



Set the state of the sampler using a sample of visible vectors.<br /><br />Notes:<br /> ~ Modifies the sampler.state attribute in place.<br /><br />Args:<br /> ~ batch: a Batch object<br /><br />Returns:<br /> ~ None


### state\_for\_grad
```py

def state_for_grad(self, target_layer, dropout_mask=None)

```



Peform a mean field update of the target layer.<br /><br />Args:<br /> ~ target_layer (int): the layer to update<br /> ~ dropout_mask (State): mask on model units<br /><br />Returns:<br /> ~ state


### update\_state
```py

def update_state(self, steps, dropout_mask=None)

```



Update the state of the particles.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /> ~ Calls _update_beta() method.<br /><br />Args:<br /> ~ steps (int): the number of Monte Carlo steps<br /> ~ dropout_mask (State object): mask on model units<br /> ~  ~ for positive phase dropout, 1: on 0: dropped-out<br /><br />Returns:<br /> ~ None




## class LayerwisePretrain
Pretrain a model in layerwise fashion using the method from:<br /><br />"Deep Boltzmann Machines" by Ruslan Salakhutdinov and Geoffrey Hinton
### \_\_init\_\_
```py

def __init__(self, model, batch, optimizer, epochs, method=<function persistent_contrastive_divergence at 0x11e805950>, mcsteps=1, metrics=None)

```



Create a LayerwisePretrain object.<br /><br />Args:<br /> ~ model: a model object<br /> ~ batch: a batch object<br /> ~ optimizer: an optimizer object<br /> ~ epochs (int): the number of epochs<br /> ~ method (optional): the method used to approximate the likelihood<br /> ~  ~  ~  ~  ~    gradient [cd, pcd, tap]<br /> ~ mcsteps (int, optional): the number of Monte Carlo steps per gradient<br /> ~ metrics (List, optional): a list of metrics<br /><br />Returns:<br /> ~ LayerwisePretrain


### train
```py

def train(self)

```



Train the model layerwise.<br /><br />Notes:<br /> ~ Updates the model parameters in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None




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

def __init__(self, model, clamped=None, updater='markov_chain')

```



Create a sequential Monte Carlo sampler.<br /><br />Args:<br /> ~ model: a model object<br /> ~ method (str; optional): how to update the particles<br /><br />Returns:<br /> ~ SequentialMC


### reset
```py

def reset(self)

```



Reset the sampler state.<br /><br />Notes:<br /> ~ Modifies sampler.state attribute in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### set\_state
```py

def set_state(self, state)

```



Set the state.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ state (State): The state of the units.<br /><br />Returns:<br /> ~ None


### set\_state\_from\_batch
```py

def set_state_from_batch(self, batch)

```



Set the state of the sampler using a sample of visible vectors.<br /><br />Notes:<br /> ~ Modifies the sampler.state attribute in place.<br /><br />Args:<br /> ~ batch: a Batch object<br /><br />Returns:<br /> ~ None


### state\_for\_grad
```py

def state_for_grad(self, target_layer, dropout_mask=None)

```



Peform a mean field update of the target layer.<br /><br />Args:<br /> ~ target_layer (int): the layer to update<br /> ~ dropout_mask (State): mask on model units<br /><br />Returns:<br /> ~ state


### update\_state
```py

def update_state(self, steps, dropout_mask=None)

```



Update the positive state of the particles.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ steps (int): the number of Monte Carlo steps<br /> ~ dropout_mask (optioonal; State): mask on model units; 1=on, 0=dropped<br /><br />Returns:<br /> ~ None




## class OrderedDict
Dictionary that remembers insertion order


## class Sampler
Base class for the sequential Monte Carlo samplers
### \_\_init\_\_
```py

def __init__(self, model, updater='markov_chain', **kwargs)

```



Create a sampler.<br /><br />Args:<br /> ~ model: a model object<br /> ~ kwargs (optional)<br /><br />Returns:<br /> ~ sampler


### set\_state
```py

def set_state(self, state)

```



Set the state.<br /><br />Notes:<br /> ~ Modifies the state attribute in place.<br /><br />Args:<br /> ~ state (State): The state of the units.<br /><br />Returns:<br /> ~ None


### set\_state\_from\_batch
```py

def set_state_from_batch(self, batch)

```



Set the state of the sampler using a sample of visible vectors.<br /><br />Notes:<br /> ~ Modifies the sampler.state attribute in place.<br /><br />Args:<br /> ~ batch: a Batch object<br /><br />Returns:<br /> ~ None




## class Model
General model class.<br />(i.e., Restricted Boltzmann Machines).<br /><br />Example usage:<br />'''<br />vis = BernoulliLayer(nvis)<br />hid = BernoulliLayer(nhid)<br />rbm = Model([vis, hid])<br />'''
### TAP\_gradient
```py

def TAP_gradient(self, data_state, init_lr, tolerance, max_iters, positive_dropout=None)

```



Gradient of -\ln P(v) with respect to the model parameters<br /><br />Args:<br /> ~ data_state (State object): The observed visible units and sampled<br /> ~  hidden units.<br /> ~ init_lr float: initial learning rate which is halved whenever necessary<br /> ~  to enforce descent.<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters (int): maximum gradient decsent steps<br /> ~ positive_dropout (State object): mask on model units for positive phase dropout<br /> ~  1: on 0: dropped-out<br /><br />Returns:<br /> ~ Gradient (namedtuple): containing gradients of the model parameters.


### \_\_init\_\_
```py

def __init__(self, layer_list: List, weight_list: List=None)

```



Create a model.<br /><br />Args:<br /> ~ layer_list: A list of layers objects.<br /><br />Returns:<br /> ~ model: A model.


### compute\_StateTAP
```py

def compute_StateTAP(self, init_lr=0.1, tol=1e-07, max_iters=50)

```



Compute the state of the layers by minimizing the second order TAP<br />approximation to the Helmholtz free energy.<br /><br />If the energy is,<br />'''<br />E(v, h) := -\langle a,v angle - \langle b,h angle - \langle v,W \cdot h angle,<br />'''<br />with Boltzmann probability distribution,<br />'''<br />P(v,h) := Z^{-1} \exp{-E(v,h)},<br />'''<br />and the marginal,<br />'''<br />P(v) := \sum_{h} P(v,h),<br />'''<br />then the Helmholtz free energy is,<br />'''<br />F(v) := - log Z = -log \sum_{v,h} \exp{-E(v,h)}.<br />'''<br />We add an auxiliary local field q, and introduce the inverse temperature<br /> variable eta to define<br />'''<br />eta F(v;q) := -log\sum_{v,h} \exp{-eta E(v,h) + eta \langle q, v angle}<br />'''<br />Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q,<br /> the Gibbs free energy.<br />The TAP formula is Taylor series of \Gamma in eta, around eta=0.<br />Setting eta=1 and regarding the first two terms of the series as an<br /> approximation of \Gamma[m],<br />we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)<br /><br />This implementation uses gradient descent from a random starting location<br /> to minimize the function<br /><br />Args:<br /> ~ init_lr float: initial learning rate<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters: maximum gradient decsent steps.<br /><br />Returns:<br /> ~ state of the layers (StateTAP)


### deterministic\_iteration
```py

def deterministic_iteration(self, n: int, state: paysage.models.model_utils.State, dropout_mask: paysage.models.model_utils.State=None, beta=None) -> paysage.models.model_utils.State

```



Perform multiple deterministic (maximum probability) updates<br />in alternating layers.<br />state -> new state<br /><br />Notes:<br /> ~ Returns the layer units that maximize the probability<br /> ~ conditioned on adjacent layers,<br /> ~ x_i = argmax P(x_i | x_(i-1), x_(i+1))<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ dropout_mask (State object):<br /> ~  ~ mask on model units for dropout, 1: on 0: dropped-out<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /><br />Returns:<br /> ~ new state


### get\_config
```py

def get_config(self) -> dict

```



Get a configuration for the model.<br /><br />Notes:<br /> ~ Includes metadata on the layers.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the model.


### gibbs\_free\_energy
```py

def gibbs_free_energy(self, state)

```



Gibbs Free Energy (GFE) according to TAP2 appoximation<br /><br />Args:<br /> ~ state (StateTAP): cumulants of the layers<br /><br />Returns:<br /> ~ float: Gibbs free energy


### grad\_TAP\_free\_energy
```py

def grad_TAP_free_energy(self, init_lr_EMF, tolerance_EMF, max_iters_EMF)

```



Compute the gradient of the Helmholtz free engergy of the model according<br />to the TAP expansion around infinite temperature.<br /><br />This function will use the class members which specify the parameters for<br />the Gibbs FE minimization.<br />The gradients are taken as the average over the gradients computed at<br />each of the minimial magnetizations for the Gibbs FE.<br /><br />Args:<br /> ~ init_lr float: initial learning rate which is halved whenever necessary<br /> ~ to enforce descent.<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters int: maximum gradient decsent steps<br /><br />Returns:<br /> ~ namedtuple: (Gradient): containing gradients of the model parameters.


### gradient
```py

def gradient(self, data_state, model_state, positive_dropout=None, negative_dropout=None)

```



Compute the gradient of the model parameters.<br />Scales the units in the state and computes the gradient.<br /><br />Args:<br /> ~ data_state (State object): The observed visible units and sampled hidden units.<br /> ~ model_state (State object): The visible and hidden units sampled from the model.<br /> ~ positive_dropout (State object): mask on model units<br /> ~  ~ for positive phase dropout, 1: on 0: dropped-out<br /> ~ negative_dropout (State object): mask on model units<br /> ~  ~ for negative phase dropout, 1: on 0: dropped-out<br /><br />Returns:<br /> ~ dict: Gradients of the model parameters.


### initialize
```py

def initialize(self, data, method: str='hinton') -> None

```



Initialize the parameters of the model.<br /><br />Args:<br /> ~ data: A Batch object.<br /> ~ method (optional): The initialization method.<br /><br />Returns:<br /> ~ None


### joint\_energy
```py

def joint_energy(self, data)

```



Compute the joint energy of the model based on a state.<br /><br />Args:<br /> ~ data (State object): the current state of each layer<br /><br />Returns:<br /> ~ tensor (num_samples,): Joint energies.


### markov\_chain
```py

def markov_chain(self, n: int, state: paysage.models.model_utils.State, dropout_mask: paysage.models.model_utils.State=None, beta=None) -> paysage.models.model_utils.State

```



Perform multiple Gibbs sampling steps in alternating layers.<br />state -> new state<br /><br />Notes:<br /> ~ Samples layers according to the conditional probability<br /> ~ on adjacent layers,<br /> ~ x_i ~ P(x_i | x_(i-1), x_(i+1) )<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ dropout_mask (State object):<br /> ~  ~ mask on model units for dropout, 1: on 0: dropped-out<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /><br />Returns:<br /> ~ new state


### mean\_field\_iteration
```py

def mean_field_iteration(self, n: int, state: paysage.models.model_utils.State, dropout_mask: paysage.models.model_utils.State=None, beta=None) -> paysage.models.model_utils.State

```



Perform multiple mean-field updates in alternating layers<br />states -> new state<br /><br />Notes:<br /> ~ Returns the expectation of layer units<br /> ~ conditioned on adjacent layers,<br /> ~ x_i = E[x_i | x_(i-1), x_(i+1) ]<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ dropout_mask (State object):<br /> ~  ~ mask on model units for dropout, 1: on 0: dropped-out<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /><br />Returns:<br /> ~ new state


### parameter\_update
```py

def parameter_update(self, deltas)

```



Update the model parameters.<br /><br />Notes:<br /> ~ Modifies the model parameters in place.<br /><br />Args:<br /> ~ deltas (Gradient)<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, vis)

```



Generate a random sample with the same shape,<br />and of the same type, as the visible units.<br /><br />Args:<br /> ~ vis: The visible units.<br /><br />Returns:<br /> ~ tensor: Random sample with same shape as vis.


### save
```py

def save(self, store: pandas.io.pytables.HDFStore) -> None

```



Save a model to an open HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore)<br /><br />Returns:<br /> ~ None


### use\_dropout
```py

def use_dropout(self)

```



Indicate if the model has dropout.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ true of false




## class State
A State is a list of tensors that contains the states of the units<br />described by a model.<br /><br />For a model with L hidden layers, the tensors have shapes<br /><br />shapes = [<br />(num_samples, num_visible),<br />(num_samples, num_hidden_1),<br />            .<br />            .<br />            .<br />(num_samples, num_hidden_L)<br />]
### \_\_init\_\_
```py

def __init__(self, tensors)

```



Create a State object.<br /><br />Args:<br /> ~ tensors: a list of tensors<br /><br />Returns:<br /> ~ state object




## functions

### cd
```py

def cd(vdata, model, positive_phase, negative_phase, positive_dropout=None, steps=1)

```



Compute an approximation to the likelihood gradient using the CD-k<br />algorithm for approximate maximum likelihood inference.<br /><br />Hinton, Geoffrey E.<br />"Training products of experts by minimizing contrastive divergence."<br />Neural computation 14.8 (2002): 1771-1800.<br /><br />Carreira-Perpinan, Miguel A., and Geoffrey Hinton.<br />"On Contrastive Divergence Learning."<br />AISTATS. Vol. 10. 2005.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /> ~ Modifies the sampling attributes of the model's compute graph.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ positive_phase: a sampler object<br /> ~ negative_phase: a sampler object<br /> ~ positive_dropout (State object): mask on model units<br /> ~  for dropout 1: on 0: dropped out<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### contrastive\_divergence
```py

def contrastive_divergence(vdata, model, positive_phase, negative_phase, positive_dropout=None, steps=1)

```



Compute an approximation to the likelihood gradient using the CD-k<br />algorithm for approximate maximum likelihood inference.<br /><br />Hinton, Geoffrey E.<br />"Training products of experts by minimizing contrastive divergence."<br />Neural computation 14.8 (2002): 1771-1800.<br /><br />Carreira-Perpinan, Miguel A., and Geoffrey Hinton.<br />"On Contrastive Divergence Learning."<br />AISTATS. Vol. 10. 2005.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /> ~ Modifies the sampling attributes of the model's compute graph.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ positive_phase: a sampler object<br /> ~ negative_phase: a sampler object<br /> ~ positive_dropout (State object): mask on model units<br /> ~  for dropout 1: on 0: dropped out<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### pcd
```py

def pcd(vdata, model, positive_phase, negative_phase, positive_dropout=None, steps=1)

```



PCD-k algorithm for approximate maximum likelihood inference.<br /><br />Tieleman, Tijmen.<br />"Training restricted Boltzmann machines using approximations to the<br />likelihood gradient."<br />Proceedings of the 25th international conference on Machine learning.<br />ACM, 2008.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /> ~ Modifies the sampling attributes of the model's compute graph.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ positive_phase: a sampler object<br /> ~ negative_phase: a sampler object<br /> ~ positive_dropout (State object): mask on model units for positive phase dropout<br /> ~  1: on 0: dropped-out<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### persistent\_contrastive\_divergence
```py

def persistent_contrastive_divergence(vdata, model, positive_phase, negative_phase, positive_dropout=None, steps=1)

```



PCD-k algorithm for approximate maximum likelihood inference.<br /><br />Tieleman, Tijmen.<br />"Training restricted Boltzmann machines using approximations to the<br />likelihood gradient."<br />Proceedings of the 25th international conference on Machine learning.<br />ACM, 2008.<br /><br />Notes:<br /> ~ Modifies the state of the sampler.<br /> ~ Modifies the sampling attributes of the model's compute graph.<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ positive_phase: a sampler object<br /> ~ negative_phase: a sampler object<br /> ~ positive_dropout (State object): mask on model units for positive phase dropout<br /> ~  1: on 0: dropped-out<br /> ~ steps (int): the number of Monte Carlo steps<br /><br />Returns:<br /> ~ gradient


### tap
```py

def tap(vdata, model, positive_phase, negative_phase=None, steps=1, init_lr_EMF=0.1, tolerance_EMF=0.0001, max_iters_EMF=25, positive_dropout=None)

```



Compute the gradient using the Thouless-Anderson-Palmer (TAP)<br />mean field approximation.<br /><br />Slight modifications on the methods in<br /><br />Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,<br />and Florent Krzakala<br />"A Deterministic and Generalized Framework for Unsupervised Learning<br />with Restricted Boltzmann Machines"<br /><br />Args:<br /> ~ vdata (tensor): observed visible units<br /> ~ model: a model object<br /> ~ positive_phase: a sampler object<br /> ~ negative_phase (unused; default=None): a sampler object<br /> ~ steps: steps to sample MCMC for positive phase<br /> ~ positive_dropout (State object): mask on model units for positive phase dropout<br /> ~  1: on 0: dropped-out<br /><br /> ~ TAP free energy computation parameters:<br /> ~  ~ init_lr float: initial learning rate which is halved whenever necessary to enforce descent.<br /> ~  ~ tol float: tolerance for quitting minimization.<br /> ~  ~ max_iters: maximum gradient decsent steps<br /><br />Returns:<br /> ~ gradient object

