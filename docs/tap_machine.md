# Documentation for Tap_Machine (tap_machine.py)

## class TAP_rbm
RBM with TAP formula-based gradient which supports deterministic training<br /><br />Example usage:<br />'''<br />vis = BernoulliLayer(nvis)<br />hid = BernoulliLayer(nhid)<br />rbm = TAP_rbm([vis, hid])<br />'''
### \_\_init\_\_
```py

def __init__(self, layer_list)

```



Create a model.<br /><br />Notes:<br /> ~ Only 2-layer models currently supported.<br /><br />Args:<br /> ~ layer_list: A list of layers objects.<br /><br />Returns:<br /> ~ model: A model.


### clamped\_free\_energy
```py

def clamped_free_energy(self, v, w, a, b)

```



'''<br />-\log \sum_h \exp{-E(v,h)}<br />'''


### deterministic\_iteration
```py

def deterministic_iteration(self, vis, n: int, beta=None)

```



Perform multiple deterministic (maximum probability) updates.<br />v -> h -> v_1 -> h_1 -> ... -> v_n<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ n: Number of steps.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


### deterministic\_step
```py

def deterministic_step(self, vis, beta=None)

```



Perform a single deterministic (maximum probability) update.<br />v -> update h distribution -> h -> update v distribution -> v'<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


### get\_config
```py

def get_config(self) -> dict

```



Get a configuration for the model.<br /><br />Notes:<br /> ~ Includes metadata on the layers.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the model.


### gibbs\_free\_energy\_TAP2
```py

def gibbs_free_energy_TAP2(self, init_lr=0.1, tol=0.0001, max_iters=500)

```



Compute the Gibbs free engergy of the model according to the TAP<br />expansion around infinite temperature to second order.<br /><br />If the energy is:<br />'''<br /> ~ E(v, h) := -\langle a,v angle - \langle b,h angle - \langle v,W \cdot h angle, with state probability distribution:<br /> ~ P(v,h)  := 1/\sum_{v,h} \exp{-E(v,h)} * \exp{-E(v,h)}, and the marginal<br /> ~ P(v) ~ := \sum_{h} P(v,h)<br />'''<br />Then the Gibbs free energy is:<br />'''<br /> ~ F(v) := -log\sum_{v,h} \exp{-E(v,h)}<br />'''<br />We add an auxiliary local field q, and introduce the inverse temperature variable eta to define<br />'''<br /> ~ eta F(v;q) := -log\sum_{v,h} \exp{-eta E(v,h) + eta \langle q, v angle}<br />'''<br />Let \Gamma(m) be the Legendre transform of F(v;q) as a function of q<br />The TAP formula is Taylor series of \Gamma in eta, around eta=0.<br />Setting eta=1 and regarding the first two terms of the series as an approximation of \Gamma[m],<br />we can minimize \Gamma in m to obtain an approximation of F(v;q=0) = F(v)<br /><br />This implementation uses gradient descent from a random starting location to minimize the function<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ init_lr float: initial learning rate which is halved whenever necessary to enforce descent.<br /> ~ tol float: tolerance for quitting minimization.<br /> ~ max_iters: maximum gradient decsent steps<br /><br />Returns:<br /> ~ tuple (visible magnetization, hidden magnetization, TAP2-approximated Gibbs free energy)<br /> ~ (num_visible_neurons, num_hidden_neurons, 1)


### gradient
```py

def gradient(self, vdata, vmodel)

```



Gradient of -\ln P(v) with respect to the weights and biases


### initialize
```py

def initialize(self, data, method: str='hinton')

```



Inialize the parameters of the model.<br /><br />Args:<br /> ~ data: A batch object.<br /> ~ method (optional): The initalization method.<br /><br />Returns:<br /> ~ None


### joint\_energy
```py

def joint_energy(self, vis, hid)

```



Compute the joint energy of the model.<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ hid (batch_size, num_hidden): Sampled hidden units:<br /><br />Returns:<br /> ~ tensor (batch_size, ): Joint energies.


### marginal\_free\_energy
```py

def marginal_free_energy(self, vis)

```



Compute the marginal free energy of the model.<br /><br />If the energy is:<br />E(v, h) = -\sum_i a_i(v_i) - \sum_j b_j(h_j) - \sum_{ij} W_{ij} v_i h_j<br />Then the marginal free energy is:<br />F(v) =  -\sum_i a_i(v_i) - \sum_j \log \int dh_j \exp(b_j(h_j) - \sum_i W_{ij} v_i)<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /><br />Returns:<br /> ~ tensor (batch_size, ): Marginal free energies.


### markov\_chain
```py

def markov_chain(self, vis, n, beta=None)

```



Perform multiple Gibbs sampling steps.<br />v ~ h ~ v_1 ~ h_1 ~ ... ~ v_n<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ n: Number of steps.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


### mcstep
```py

def mcstep(self, vis, beta=None)

```



Perform a single Gibbs sampling update.<br />v -> update h distribution ~ h -> update v distribution ~ v'<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


### mean\_field\_iteration
```py

def mean_field_iteration(self, vis, n, beta=None)

```



Perform multiple mean-field updates.<br />v -> h -> v_1 -> h_1 -> ... -> v_n<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ n: Number of steps.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


### mean\_field\_step
```py

def mean_field_step(self, vis, beta=None)

```



Perform a single mean-field update.<br />v -> update h distribution -> h -> update v distribution -> v'<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


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



Save a model to an open HDFStore.<br /><br />Note:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore)<br /><br />Returns:<br /> ~ None



