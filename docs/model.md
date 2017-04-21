# Documentation for Model (model.py)

## class Model
General model class.<br />Currently only supports models with 2 layers,<br />(i.e., Restricted Boltzmann Machines).<br /><br />Example usage:<br />'''<br />vis = BernoulliLayer(nvis)<br />hid = BernoulliLayer(nhid)<br />rbm = Model([vis, hid])<br />'''
### \_\_init\_\_
```py

def __init__(self, layer_list)

```



Create a model.<br /><br />Notes:<br /> ~ Only 2-layer models currently supported.<br /><br />Args:<br /> ~ layer_list: A list of layers objects.<br /><br />Returns:<br /> ~ model: A model.


### deterministic\_iteration
```py

def deterministic_iteration(self, n, state, beta=None, clamped=[])

```



Perform multiple deterministic (maximum probability) updates<br />in alternating layers.<br />state -> new state<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /> ~ clamped (list): list of layer indices to clamp<br /><br />Returns:<br /> ~ new state


### get\_config
```py

def get_config(self) -> dict

```



Get a configuration for the model.<br /><br />Notes:<br /> ~ Includes metadata on the layers.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the model.


### gradient
```py

def gradient(self, data_state, model_state)

```



Compute the gradient of the model parameters.<br /><br />For vis \in {vdata, vmodel}, we:<br /><br />1. Scale the visible data.<br />vis_scaled = self.layers[i].rescale(vis)<br /><br />2. Update the hidden layer.<br />self.layers[i+1].update(vis_scaled, self.weights[i].W())<br /><br />3. Compute the mean of the hidden layer.<br />hid = self.layers[i].mean()<br /><br />4. Scale the mean of the hidden layer.<br />hid_scaled = self.layers[i+1].rescale(hid)<br /><br />5. Compute the derivatives.<br />vis_derivs = self.layers[i].derivatives(vis, hid_scaled,<br /> ~  ~  ~  ~  ~  ~  ~  ~  ~  ~ self.weights[i].W())<br />hid_derivs = self.layers[i+1].derivatives(hid, vis_scaled,<br /> ~  ~  ~  ~  ~  ~  ~   be.transpose(self.weights[i+1].W())<br />weight_derivs = self.weights[i].derivatives(vis_scaled, hid_scaled)<br /><br />The gradient is obtained by subtracting the vmodel contribution<br />from the vdata contribution.<br /><br />Args:<br /> ~ data_state (State object): The observed visible units and sampled hidden units.<br /> ~ model_state (State objects): The visible and hidden units sampled from the model.<br /><br />Returns:<br /> ~ dict: Gradients of the model parameters.


### initialize
```py

def initialize(self, data, method: str='hinton')

```



Inialize the parameters of the model.<br /><br />Args:<br /> ~ data: A Batch object.<br /> ~ method (optional): The initalization method.<br /><br />Returns:<br /> ~ None


### joint\_energy
```py

def joint_energy(self, data)

```



Compute the joint energy of the model based on a state.<br /><br />Args:<br /> ~ data (State object): the current state of each layer<br /><br />Returns:<br /> ~ tensor (num_samples,): Joint energies.


### marginal\_free\_energy
```py

def marginal_free_energy(self, data)

```



Compute the marginal free energy of the model.<br /><br />If the energy is:<br />E(v, h) = -\sum_i a_i(v_i) - \sum_j b_j(h_j) - \sum_{ij} W_{ij} v_i h_j<br />Then the marginal free energy is:<br />F(v) =  -\sum_i a_i(v_i) - \sum_j \log \int dh_j \exp(b_j(h_j) - \sum_i W_{ij} v_i)<br />This can be extended to a deep model by a sum over all hidden states<br /><br />Args:<br /> ~ data (State object): The current state of each layer.<br /><br />Returns:<br /> ~ tensor (batch_size, ): Marginal free energies.


### markov\_chain
```py

def markov_chain(self, n, state, beta=None, clamped=[])

```



Perform multiple Gibbs sampling steps in alternating layers.<br />state -> new state<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /> ~ clamped (list): list of layer indices to clamp<br /><br />Returns:<br /> ~ new state


### mean\_field\_iteration
```py

def mean_field_iteration(self, n, state, beta=None, clamped=[])

```



Perform multiple mean-field updates in alternating layers<br />states -> new state<br /><br />Args:<br /> ~ n (int): number of steps.<br /> ~ state (State object): the current state of each layer<br /> ~ beta (optional, tensor (batch_size, 1)): Inverse temperatures<br /> ~ clamped (list): list of layer indices to clamp<br /><br />Returns:<br /> ~ new state


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




## class State
A State is a list of tensors that contains the states of the units<br />described by a model.<br /><br />For a model with L hidden layers, the tensors have shapes<br /><br />shapes = [<br />(num_samples, num_visible),<br />(num_samples, num_hidden_1),<br />            .<br />            .<br />            .<br />(num_samples, num_hidden_L)<br />]
### \_\_init\_\_
```py

def __init__(self, tensors)

```



Create a State object.<br /><br />Args:<br /> ~ tensors: a list of tensors<br /><br />Returns:<br /> ~ state object



