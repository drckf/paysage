# Documentation for Layer (layer.py)

## class CumulantsTAP
CumulantsTAP(mean, variance)


## class OrderedDict
Dictionary that remembers insertion order


## class ParamsLayer
Params()


## class Layer
A general layer class with common functionality.
### \_\_init\_\_
```py

def __init__(self, num_units, dropout_p, *args, **kwargs)

```



Basic layer initialization method.<br /><br />Args:<br /> ~ num_units (int): number of units in the layer<br /> ~ dropout_p (float): likelihood each unit is dropped out in training<br /> ~ *args: any arguments<br /> ~ **kwargs: any keyword arguments<br /><br />Returns:<br /> ~ layer


### add\_constraint
```py

def add_constraint(self, constraint)

```



Add a parameter constraint to the layer.<br /><br />Notes:<br /> ~ Modifies the layer.contraints attribute in place.<br /><br />Args:<br /> ~ constraint (dict): {param_name: constraint (paysage.constraints)}<br /><br />Returns:<br /> ~ None


### add\_penalty
```py

def add_penalty(self, penalty)

```



Add a penalty to the layer.<br /><br />Note:<br /> ~ Modfies the layer.penalties attribute in place.<br /><br />Args:<br /> ~ penalty (dict): {param_name: penalty (paysage.penalties)}<br /><br />Returns:<br /> ~ None


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### get\_base\_config
```py

def get_base_config(self)

```



Get a base configuration for the layer.<br /><br />Notes:<br /> ~ Encodes metadata for the layer.<br /> ~ Includes the base layer data.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the layer.


### get\_config
```py

def get_config(self)

```



Get a full configuration for the layer.<br /><br />Notes:<br /> ~ Encodes metadata on the layer.<br /> ~ Weights are separately retrieved.<br /> ~ Builds the base configuration.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the layer.


### get\_dropout\_mask
```py

def get_dropout_mask(self, batch_size=1)

```



Return a binary mask<br /><br />Args:<br /> ~ batch_size (int): number of masks to generate<br /><br />Returns:<br /> ~ mask (tensor (batch_size, self.len): binary mask


### get\_dropout\_scale
```py

def get_dropout_scale(self, batch_size=1)

```



Return a vector representing the probability that each unit is on<br /> ~ with respect to dropout<br /><br />Args:<br /> ~ batch_size (int): number of copies to return<br /><br />Returns:<br /> ~ scale (tensor (batch_size, self.len)): vector of scales for each unit


### get\_penalties
```py

def get_penalties(self)

```



Get the value of the penalties:<br /><br />E.g., L2 penalty = (1/2) * penalty * \sum_i parameter_i ** 2<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ dict (float): the values of the penalty functions


### get\_penalty\_grad
```py

def get_penalty_grad(self, deriv, param_name)

```



Get the gradient of the penalties on a parameter.<br /><br />E.g., L2 penalty gradient = penalty * parameter_i<br /><br />Args:<br /> ~ deriv (tensor): derivative of the parameter<br /> ~ param_name: name of the parameter<br /><br />Returns:<br /> ~ tensor: derivative including penalty


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the readable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (List[namedtuple]): List[param_name: tensor] (update)<br /><br />Returns:<br /> ~ None


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### set\_fixed\_params
```py

def set_fixed_params(self, fixed_params)

```



Set the params that are not trainable.<br /><br />Notes:<br /> ~ Modifies the layer.fixed_params attribute in place.<br /><br />Args:<br /> ~ fixed_params (List[str]): a list of the fixed parameter names<br /><br />Returns:<br /> ~ None


### set\_params
```py

def set_params(self, new_params)

```



Set the value of the layer params attribute.<br /><br />Notes:<br /> ~ Modifies layer.params in place.<br /><br />Args:<br /> ~ new_params (namedtuple)<br /><br />Returns:<br /> ~ None


### use\_dropout
```py

def use_dropout(self)

```



Indicate if the layer has dropout.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ true of false




## functions

### namedtuple
```py

def namedtuple(typename, field_names, *, verbose=False, rename=False, module=None)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)

