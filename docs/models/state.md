# Documentation for State (state.py)

## class StateTAP
A StateTAP is a list of CumulantsTAP objects for each layer in the model.
### \_\_init\_\_
```py

def __init__(self, cumulants, lagrange_multipliers)

```



Create a StateTAP.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants: list of CumulantsTAP objects<br />&nbsp;&nbsp;&nbsp;&nbsp;lagrange_multipliers: list of CumulantsTAP objects<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;StateTAP




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




## functions

### state\_allclose
```py

def state_allclose(state1: paysage.models.state.State, state2: paysage.models.state.State, rtol: float=1e-05, atol: float=1e-08) -> bool

```



Check that for approximate equality between two states.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;state1 (State): a state to compare.<br />&nbsp;&nbsp;&nbsp;&nbsp;state2 (State): a state to compare.<br />&nbsp;&nbsp;&nbsp;&nbsp;rtol (optional): Relative tolerance.<br />&nbsp;&nbsp;&nbsp;&nbsp;atol (optional): Absolute tolerance.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;approximate equality (bool)

