# Documentation for Model (model.py)

## class partial
partial(func, *args, **keywords) - new function with partial application<br />of the given arguments and keywords.


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




## class List
list() -> new empty list<br />list(iterable) -> new list initialized from iterable's items

