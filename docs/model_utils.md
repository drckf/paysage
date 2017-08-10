# Documentation for Model_Utils (model_utils.py)

## class ComputationGraph
Manages the connections between layers in a model.<br />Layers can have various properties set:<br />    - Clamped sampling: the layer is not sampled (state unchanged)<br />    - Trainable: the layer's parameters can be changed<br />Weight layers (connecting two other layers) can have<br />    the trainable property set.<br /><br />The computation graph is defined by an Graph object, where<br />layers are vertices and weight layers are edges.<br />The incidence matrix defines the connections between layers<br />and the corresponding edges.
### \_\_init\_\_
```py

def __init__(self, num_layers)

```



Builds the default incidence matrix.<br /><br />Args:<br /> ~ num_layers: the number of layers in the model<br /><br />Returns:<br /> ~ None


### default\_incidence\_matrix
```py

def default_incidence_matrix(self, num_layers)

```



Builds the default incidence matrix.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ incidence_matrix: a matrix specifying the connections in the model.


### get\_layer\_connections
```py

def get_layer_connections(self)

```



Returns a list over layers.<br />Each element of the list is a list of LayerConnection tuples,<br />which are the connections to that layer.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ layer_connections (List): a list over layers, where each entry is<br /> ~  ~ a list of LayerConnection tuples of the connections to that layer.


### get\_sampled
```py

def get_sampled(self)

```



Convenience function that returns the layers for which sampling is<br />not clamped.<br />Complement of the `clamped_sampling` attribute.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ unclamped_sampling (List): layers for which sampling is not clamped.


### set\_clamped\_sampling
```py

def set_clamped_sampling(self, clamped_sampling)

```



Convenience function to set the layers for which sampling is clamped.<br />Sets exactly the given layers to have sampling clamped.<br /><br />Args:<br /> ~ clamped_sampling (List): the exact set of layers which are have sampling clamped.<br /><br />Returns:<br /> ~ None


### set\_trainable\_layers
```py

def set_trainable_layers(self, trainable_layers)

```



Convenience function to set the layers which are trainable<br />Sets exactly the given layers as trainable.<br />Sets weights untrainable if the higher index layer is untrainable.<br /><br />Args:<br /> ~ trainable_layers (List): the exact set of layers which are trainable<br /><br />Returns:<br /> ~ None




## class LayerConnection
LayerConnection(layer, weight, is_forward)


## class StateTAP
A StateTAP is a list of CumulantsTAP objects for each layer in the model.
### \_\_init\_\_
```py

def __init__(self, cumulants)

```



Create a StateTAP.<br /><br />Args:<br /> ~ cumulants: list of CumulantsTAP objects<br /><br />Returns:<br /> ~ StateTAP




## class Graph
Container for the contents of a graph.<br />Allows operations to extract useful objects,<br />    such as the adjacency matrix, adjacency list, and edge list.
### \_\_init\_\_
```py

def __init__(self, incidence_matrix)

```



Initialize self.  See help(type(self)) for accurate signature.




## class State
A State is a list of tensors that contains the states of the units<br />described by a model.<br /><br />For a model with L hidden layers, the tensors have shapes<br /><br />shapes = [<br />(num_samples, num_visible),<br />(num_samples, num_hidden_1),<br />            .<br />            .<br />            .<br />(num_samples, num_hidden_L)<br />]
### \_\_init\_\_
```py

def __init__(self, tensors)

```



Create a State object.<br /><br />Args:<br /> ~ tensors: a list of tensors<br /><br />Returns:<br /> ~ state object




## functions

### deepcopy
```py

def deepcopy(x, memo=None, _nil=[])

```



Deep copy operation on arbitrary Python objects.<br /><br />See the module's __doc__ string for more info.


### dropout\_state
```py

def dropout_state(state: paysage.models.model_utils.State, dropout_mask: paysage.models.model_utils.State) -> paysage.models.model_utils.State

```



Apply a dropout mask to a state.<br /><br />Args:<br /> ~ state (State): the state of the units<br /> ~ dropout_mask (State): the dropout mask<br /><br />Returns:<br /> ~ new state


### dropout\_state\_
```py

def dropout_state_(state: paysage.models.model_utils.State, dropout_mask: paysage.models.model_utils.State) -> None

```



Apply a dropout mask to a state.<br /><br />Notes:<br /> ~ Changes state in place!<br /><br />Args:<br /> ~ state (State): the state of the units<br /> ~ dropout_mask (State): the dropout mask<br /><br />Returns:<br /> ~ None


### namedtuple
```py

def namedtuple(typename, field_names, *, verbose=False, rename=False, module=None)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)

