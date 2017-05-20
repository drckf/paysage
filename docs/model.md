# Documentation for Model (model.py)

## class partial
partial(func, *args, **keywords) - new function with partial application<br />of the given arguments and keywords.


## class Model
General model class.<br />(i.e., Restricted Boltzmann Machines).<br /><br />Example usage:<br />'''<br />vis = BernoulliLayer(nvis)<br />hid = BernoulliLayer(nhid)<br />rbm = Model([vis, hid])<br />'''
### TAP\_free\_energy
```py

def TAP_free_energy(self, seed=None, init_lr=0.1, tol=1e-07, max_iters=50, method='gd')

```



Compute the Helmholtz free energy of the model according to the TAP<br />expansion around infinite temperature to second order.<br /><br />If the energy is,<br />'''<br /> ~ E(v, h) := -\langle a,v angle - \langle b,h angle - \langle v,W \cdot h angle,<br />'''<br />with Boltzmann probability distribution,<br />'''<br /> ~ P(v,h)  := 1/\sum_{v,h} \exp{-E(v,h)} * \exp{-E(v,h)},<br />'''<br />and the marginal,<br />'''<br /> ~ P(v) ~ := \sum_{h} P(v,h),<br />'''<br />then the Helmholtz free energy is,<br />'''<br /> ~ F(v) := -log\sum_{v,h} \exp{-E(v,h)}.<br />'''<br />We add an auxiliary local field q, and introduce the inverse temperature variable eta to define<br />'''<br /> ~ eta F(v;q) := -log\sum_{v,h} \exp{-eta E(v,h) + eta \langle q, v angle}<br />'''<br />Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q, the Gibbs free energy.<br />The TAP formula is Taylor series of \Gamma in eta, around eta=0.<br />Setting eta=1 and regarding the first two terms of the series as an approximation of \Gamma[m],<br />we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)<br /><br />This implementation uses gradient descent from a random starting location to minimize the function<br /><br />Args:<br /> ~ seed 'None' or Magnetization: initial seed for the minimization routine.<br /> ~  ~  ~  ~  ~  ~  ~  ~   Chosing 'None' will result in a random seed<br /> ~ init_lr float: initial learning rate which is halved whenever necessary to enforce descent.<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters: maximum gradient decsent steps.<br /> ~ method: one of 'gd' or 'constraint' picking which Gibbs FE minimization method to use.<br /><br />Returns:<br /> ~ tuple (magnetization, TAP-approximated Helmholtz free energy)<br /> ~  ~   (Magnetization, float)


### TAP\_gradient
```py

def TAP_gradient(self, data_state, num_r, num_p, persistent_samples, init_lr_EMF, tolerance_EMF, max_iters_EMF)

```



Gradient of -\ln P(v) with respect to the model parameters<br /><br />Args:<br /> ~ data_state (State object): The observed visible units and sampled hidden units.<br /> ~ num_r: (int>=0) number of random seeds to use for Gibbs FE minimization<br /> ~ num_p: (int>=0) number of persistent seeds to use for Gibbs FE minimization<br /> ~ persistent_samples list of magnetizations: persistent magnetization parameters<br /> ~  ~ to keep as seeds for Gibbs free energy estimation.<br /> ~ init_lr float: initial learning rate which is halved whenever necessary to enforce descent.<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters int: maximum gradient decsent steps<br /><br />Returns:<br /> ~ namedtuple: Gradient: containing gradients of the model parameters.


### \_\_init\_\_
```py

def __init__(self, layer_list)

```



Create a model.<br /><br />Args:<br /> ~ layer_list: A list of layers objects.<br /><br />Returns:<br /> ~ model: A model.


### deterministic\_iteration
```py

def deterministic_iteration(self, n, state, beta=None, clamped=[])

```



Perform multiple deterministic (maximum probability) updates<br />in alternating layers.<br />state -> new state<br /><br />Notes:<br /> ~ Returns the layer units that maximize the probability<br /> ~ conditioned on adjacent layers,<br /> ~ x_i = argmax P(x_i | x_(i-1), x_(i+1))<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /> ~ clamped (list): list of layer indices to clamp<br /><br />Returns:<br /> ~ new state


### get\_config
```py

def get_config(self) -> dict

```



Get a configuration for the model.<br /><br />Notes:<br /> ~ Includes metadata on the layers.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the model.


### gibbs\_free\_energy
```py

def gibbs_free_energy(self, mag)

```



Gibbs FE according to TAP2 appoximation<br /><br />Args:<br /> ~ mag (list of magnetizations of layers):<br /> ~   magnetizations at which to compute the free energy<br /><br />Returns:<br /> ~ float: Gibbs free energy


### grad\_TAP\_free\_energy
```py

def grad_TAP_free_energy(self, num_r, num_p, persistent_samples, init_lr_EMF, tolerance_EMF, max_iters_EMF)

```



Compute the gradient of the Helmholtz free engergy of the model according <br />to the TAP expansion around infinite temperature.<br /><br />This function will use the class members which specify the parameters for <br />the Gibbs FE minimization.<br />The gradients are taken as the average over the gradients computed at <br />each of the minimial magnetizations for the Gibbs FE.<br /><br />Args:<br /> ~ num_r: (int>=0) number of random seeds to use for Gibbs FE minimization<br /> ~ num_p: (int>=0) number of persistent seeds to use for Gibbs FE minimization<br /> ~ persistent_samples list of magnetizations: persistent magnetization parameters<br /> ~  ~ to keep as seeds for Gibbs free energy estimation.<br /> ~ init_lr float: initial learning rate which is halved whenever necessary <br /> ~ to enforce descent.<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters int: maximum gradient decsent steps<br /><br />Returns:<br /> ~ namedtuple: (Gradient): containing gradients of the model parameters.


### gradient
```py

def gradient(self, data_state, model_state)

```



Compute the gradient of the model parameters.<br />Scales the units in the state and computes the gradient.<br /><br />Args:<br /> ~ data_state (State object): The observed visible units and sampled hidden units.<br /> ~ model_state (State objects): The visible and hidden units sampled from the model.<br /><br />Returns:<br /> ~ dict: Gradients of the model parameters.


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

def markov_chain(self, n, state, beta=None, clamped: typing.List=[]) -> paysage.models.model.State

```



Perform multiple Gibbs sampling steps in alternating layers.<br />state -> new state<br /><br />Notes:<br /> ~ Samples layers according to the conditional probability<br /> ~ on adjacent layers,<br /> ~ x_i ~ P(x_i | x_(i-1), x_(i+1) )<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /> ~ clamped (list): list of layer indices to clamp<br /><br />Returns:<br /> ~ new state


### mean\_field\_iteration
```py

def mean_field_iteration(self, n, state, beta=None, clamped=[])

```



Perform multiple mean-field updates in alternating layers<br />states -> new state<br /><br />Notes:<br /> ~ Returns the expectation of layer units<br /> ~ conditioned on adjacent layers,<br /> ~ x_i = E[x_i | x_(i-1), x_(i+1) ]<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /> ~ clamped (list): list of layer indices to clamp<br /><br />Returns:<br /> ~ new state


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




## class State
A State is a list of tensors that contains the states of the units<br />described by a model.<br /><br />For a model with L hidden layers, the tensors have shapes<br /><br />shapes = [<br />(num_samples, num_visible),<br />(num_samples, num_hidden_1),<br />            .<br />            .<br />            .<br />(num_samples, num_hidden_L)<br />]
### \_\_init\_\_
```py

def __init__(self, tensors)

```



Create a State object.<br /><br />Args:<br /> ~ tensors: a list of tensors<br /><br />Returns:<br /> ~ state object




## class List
list() -> new empty list<br />list(iterable) -> new list initialized from iterable's items


## functions

### deepcopy
```py

def deepcopy(x, memo=None, _nil=[])

```



Deep copy operation on arbitrary Python objects.<br /><br />See the module's __doc__ string for more info.

