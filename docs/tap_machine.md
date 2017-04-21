# Documentation for Tap_Machine (tap_machine.py)

## class Magnetization
### \_\_init\_\_
```py

def __init__(self, v=None, h=None)

```



Initialize self.  See help(type(self)) for accurate signature.




## class TAP_rbm
RBM with TAP formula-based gradient which supports deterministic training<br /><br />Example usage:<br />'''<br />vis = BernoulliLayer(nvis)<br />hid = BernoulliLayer(nhid)<br />rbm = TAP_rbm([vis, hid])<br />'''
### \_\_init\_\_
```py

def __init__(self, layer_list, init_lr_EMF=0.1, tolerance_EMF=0.01, max_iters_EMF=100, num_persistent_samples=0)

```



Create a TAP RBM model.<br /><br />Notes:<br /> ~ Only 2-layer models currently supported.<br /><br />Args:<br /> ~ layer_list: A list of layers objects.<br /><br /> ~ EMF computation parameters:<br /> ~  ~ init_lr float: initial learning rate which is halved whenever necessary to enforce descent.<br /> ~  ~ tol float: tolerance for quitting minimization.<br /> ~  ~ max_iters: maximum gradient decsent steps<br /> ~  ~ number of persistent magnetization parameters to keep as seeds for gradient descent.<br /> ~  ~  ~ 0 implies we use a random seed each iteration<br /><br />Returns:<br /> ~ model: A TAP RBM model.


### deterministic\_iteration
```py

def deterministic_iteration(self, n, state, beta=None, clamped=[])

```



Perform multiple deterministic (maximum probability) updates<br />in alternating layers.<br />state -> new state<br /><br />Notes:<br /> ~ Returns the layer units that maximize the probability<br /> ~ conditioned on adjacent layers,<br /> ~ x_i = argmax P(x_i | x_(i-1), x_(i+1))<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /> ~ clamped (list): list of layer indices to clamp<br /><br />Returns:<br /> ~ new state


### gamma\_TAP2
```py

def gamma_TAP2(self, m, w, a, b)

```



### get\_config
```py

def get_config(self) -> dict

```



Get a configuration for the model.<br /><br />Notes:<br /> ~ Includes metadata on the layers.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the model.


### gibbs\_free\_energy\_TAP2
```py

def gibbs_free_energy_TAP2(self, seed=None, init_lr=0.1, tol=0.0001, max_iters=500)

```



Compute the Gibbs free engergy of the model according to the TAP<br />expansion around infinite temperature to second order.<br /><br />If the energy is:<br />'''<br /> ~ E(v, h) := -\langle a,v angle - \langle b,h angle - \langle v,W \cdot h angle, with state probability distribution:<br /> ~ P(v,h)  := 1/\sum_{v,h} \exp{-E(v,h)} * \exp{-E(v,h)}, and the marginal<br /> ~ P(v) ~ := \sum_{h} P(v,h)<br />'''<br />Then the Gibbs free energy is:<br />'''<br /> ~ F(v) := -log\sum_{v,h} \exp{-E(v,h)}<br />'''<br />We add an auxiliary local field q, and introduce the inverse temperature variable eta to define<br />'''<br /> ~ eta F(v;q) := -log\sum_{v,h} \exp{-eta E(v,h) + eta \langle q, v angle}<br />'''<br />Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q<br />The TAP formula is Taylor series of \Gamma in eta, around eta=0.<br />Setting eta=1 and regarding the first two terms of the series as an approximation of \Gamma[m],<br />we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)<br /><br />This implementation uses gradient descent from a random starting location to minimize the function<br /><br />Args:<br /> ~ seed 'None' or Magnetization: initial seed for the minimization routine.<br /> ~  ~  ~  ~  ~  ~  ~  ~   Chosing 'None' will result in a random seed<br /> ~ init_lr float: initial learning rate which is halved whenever necessary to enforce descent.<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters: maximum gradient decsent steps<br /><br />Returns:<br /> ~ tuple (magnetization, TAP2-approximated Gibbs free energy)<br /> ~  ~   (Magnetization, float)


### gradient
```py

def gradient(self, data_state, model_state)

```



Gradient of -\ln P(v) with respect to the weights and biases


### initialize
```py

def initialize(self, data, method: str='hinton')

```



Initialize the parameters of the model.<br /><br />Args:<br /> ~ data: A Batch object.<br /> ~ method (optional): The initialization method.<br /><br />Returns:<br /> ~ None


### joint\_energy
```py

def joint_energy(self, data)

```



Compute the joint energy of the model based on a state.<br /><br />Args:<br /> ~ data (State object): the current state of each layer<br /><br />Returns:<br /> ~ tensor (num_samples,): Joint energies.


### marginal\_free\_energy
```py

def marginal_free_energy(self, v, w, a, b)

```



'''<br />-\log \sum_h \exp{-E(v,h)}<br />'''


### markov\_chain
```py

def markov_chain(self, n, state, beta=None, clamped=[])

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

def save(self, store)

```



Save a model to an open HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore)<br /><br />Returns:<br /> ~ None




## functions

### namedtuple
```py

def namedtuple(typename, field_names, verbose=False, rename=False)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)

