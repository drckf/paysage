# Documentation for Model_Assessment (model_assessment.py)

## class ModelAssessment
### \_\_init\_\_
```py

def __init__(self, data, model, fantasy_steps=10, num_fantasy_particles=None, beta_std=0)

```



Create a ModelAssessment object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;data (tensor ~ (num_samples, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;model (BoltzmannMachine)<br />&nbsp;&nbsp;&nbsp;&nbsp;fantasy_steps (int; optional)<br />&nbsp;&nbsp;&nbsp;&nbsp;num_fantasy_particles (int; optional)<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_std (float; optional)


### comparison
```py

def comparison(self, func, numpy=True)

```



Compare a function computed from the data and model states.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable): func: State -> tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;numpy (optional; bool): return the arrays in numpy form if true<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;data (tensor ~ 1D), model (tensor ~ 1D),<br />&nbsp;&nbsp;&nbsp;&nbsp;correlation (float), root mean squared error (float)


### sample\_data
```py

def sample_data(self, sample_indices, layer=0, func=None)

```



Select a subset samples from the data state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;sample_indices (tensor): list of indices<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (optional; int): the layer to get from the state<br />&nbsp;&nbsp;&nbsp;&nbsp;func (optional; callable): a function to apply to the units<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;data: List[tensor]


### sample\_model
```py

def sample_model(self, sample_indices, layer=0, func=None)

```



Select a subset samples from the model state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;sample_indices (tensor): list of indices<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (optional; int): the layer to get from the state<br />&nbsp;&nbsp;&nbsp;&nbsp;func (optional; callable): a function to apply to the units<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: List[tensor]


### sample\_reconstructions
```py

def sample_reconstructions(self, sample_indices, layer=0, func=None)

```



Select a subset samples from the model state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;sample_indices (tensor): list of indices<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (optional; int): the layer to get from the state<br />&nbsp;&nbsp;&nbsp;&nbsp;func (optional; callable): a function to apply to the units<br /><br />Returns:<br /><br />&nbsp;&nbsp;&nbsp;&nbsp;data: List[tensor], reconstructions: List[tensor]




## class SequentialMC
An accelerated sequential Monte Carlo sampler
### \_\_init\_\_
```py

def __init__(self, model, mcsteps=1, clamped=None, updater='markov_chain', beta_momentum=0.9, beta_std=0.6, schedule=<paysage.schedules.Constant object>)

```



Create a sequential Monte Carlo sampler.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model (BoltzmannMachine)<br />&nbsp;&nbsp;&nbsp;&nbsp;mcsteps (int; optional): the number of Monte Carlo steps<br />&nbsp;&nbsp;&nbsp;&nbsp;clamped (List[int]; optional): list of layers to clamp<br />&nbsp;&nbsp;&nbsp;&nbsp;updater (str; optional): method for updating the state<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_momentum (float in [0,1]; optional): autoregressive coefficient<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the inverse temperature of beta<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_std (float >= 0; optional): the standard deviation of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;inverse temperature beta<br />&nbsp;&nbsp;&nbsp;&nbsp;schedule (generator; optional)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;SequentialMC


### reset
```py

def reset(self)

```



Reset the sampler state.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies sampler.state attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_state
```py

def set_state(self, state)

```



Set the state.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;state (State): The state of the units.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_state\_from\_visible
```py

def set_state_from_visible(self, vdata)

```



Set the state of the sampler using a sample of visible vectors.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the sampler.state attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vdata (tensor~(num_samples,num_units)): a visible state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### state\_for\_grad
```py

def state_for_grad(self, target_layer)

```



Peform a mean field update of the target layer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;target_layer (int): the layer to update<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;state


### update\_state
```py

def update_state(self, steps=None)

```



Update the state of the particles.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state attribute in place.<br />&nbsp;&nbsp;&nbsp;&nbsp;Calls the beta_sampler.update_beta() method.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;steps (int): the number of Monte Carlo steps<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class State
A State is a list of tensors that contains the states of the units<br />described by a model.<br /><br />For a model with L hidden layers, the tensors have shapes<br /><br />shapes = [<br />(num_samples, num_visible),<br />(num_samples, num_hidden_1),<br />            .<br />            .<br />            .<br />(num_samples, num_hidden_L)<br />]
### \_\_init\_\_
```py

def __init__(self, tensors)

```



Create a State object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: a list of tensors<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;state object


### batch\_size
```py

def batch_size(self)

```



Get the batch size of the state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch size: int


### get\_visible
```py

def get_visible(self)

```



Extract the visible units<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;vis (tensor (num_samples, num_visible)): visible unit values.


### number\_of\_layers
```py

def number_of_layers(self)

```



Get the number of layers in the state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;number_of_layers (int)


### number\_of\_units
```py

def number_of_units(self, layer)

```



Get the number of units in a layer of the state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (int)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;number_of_units (int)



