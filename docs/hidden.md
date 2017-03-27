# Documentation for Hidden (hidden.py)

## class Model
General model class.<br />Currently only supports models with 2 layers,<br />(i.e., Restricted Boltzmann Machines).<br /><br />Example usage:<br />'''<br />vis = BernoulliLayer(nvis)<br />hid = BernoulliLayer(nhid)<br />rbm = Model([vis, hid])<br />'''
### \_\_init\_\_
```py

def __init__(self, layer_list)

```



Create a model.<br /><br />Notes:<br /> ~ Only 2-layer models currently supported.<br /><br />Args:<br /> ~ layer_list: A list of layers objects.<br /><br />Returns:<br /> ~ model: A model.


### deterministic\_iteration
```py

def deterministic_iteration(self, vis, n, beta=None)

```



Perform multiple deterministic (maximum probability) updates.<br />v -> h -> v_1 -> h_1 -> ... -> v_n<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ n: Number of steps.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


### deterministic\_step
```py

def deterministic_step(self, vis, beta=None)

```



Perform a single deterministic (maximum probability) update.<br />v -> update h distribution -> h -> update v distribution -> v'<br /><br />Args:<br /> ~ vis (batch_size, num_visible): Observed visible units.<br /> ~ beta (optional, (batch_size, 1)): Inverse temperatures.<br /><br />Returns:<br /> ~ tensor: New visible units (v').


### gradient
```py

def gradient(self, vdata, vmodel)

```



Compute the gradient of the model parameters.<br /><br />For vis \in {vdata, vmodel}, we:<br /><br />1. Scale the visible data.<br />vis_scaled = self.layers[i].rescale(vis)<br /><br />2. Update the hidden layer.<br />self.layers[i+1].update(vis_scaled, self.weights[i].W())<br /><br />3. Compute the mean of the hidden layer.<br />hid = self.layers[i].mean()<br /><br />4. Scale the mean of the hidden layer.<br />hid_scaled = self.layers[i+1].rescale(hid)<br /><br />5. Compute the derivatives.<br />vis_derivs = self.layers[i].derivatives(vis, hid_scaled,<br /> ~  ~  ~  ~  ~  ~  ~  ~  ~  ~ self.weights[i].W())<br />hid_derivs = self.layers[i+1].derivatives(hid, vis_scaled,<br /> ~  ~  ~  ~  ~  ~  ~   be.transpose(self.weights[i+1].W())<br />weight_derivs = self.weights[i].derivatives(vis_scaled, hid_scaled)<br /><br />The gradient is obtained by subtracting the vmodel contribution<br />from the vdata contribution.<br /><br />Args:<br /> ~ vdata: The observed visible units.<br /> ~ vmodel: The sampled visible units.<br /><br />Returns:<br /> ~ dict: Gradients of the model parameters.


### initialize
```py

def initialize(self, data, method='hinton')

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



Update the model parameters.<br /><br />Notes:<br /> ~ Modifies the model parameters in place.<br /><br />Args:<br /> ~ deltas: A dictionary of parameter updates.<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, vis)

```



Generate a random sample with the same shape,<br />and of the same type, as the visible units.<br /><br />Args:<br /> ~ vis: The visible units.<br /><br />Returns:<br /> ~ tensor: Random sample with same shape as vis.



