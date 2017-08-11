# Documentation for Weights (weights.py)

## class ParamsWeights
ParamsWeights(matrix,)


## class OrderedDict
Dictionary that remembers insertion order


## class Weights
Layer class for weights.
### GFE\_derivatives
```py

def GFE_derivatives(self, target_units, domain_units, penalize=True)

```



Gradient of the Gibbs free energy associated with this layer<br /><br />Args:<br /> ~ target_units (CumulantsTAP): magnetization of the shallower layer linked to w<br /> ~ domain_units (CumulantsTAP): magnetization of the deeper layer linked to w<br /><br />Returns:<br /> ~ derivs (namedtuple): 'matrix': tensor (contains gradient)


### W
```py

def W(self)

```



Get the weight matrix.<br /><br />A convenience method for accessing layer.params.matrix<br />with a shorter syntax.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor: weight matrix


### W\_T
```py

def W_T(self)

```



Get the transpose of the weight matrix.<br /><br />A convenience method for accessing the transpose of<br />layer.params.matrix with a shorter syntax.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor: transpose of weight matrix


### \_\_init\_\_
```py

def __init__(self, shape)

```



Create a weight layer.<br /><br />Notes:<br /> ~ The shape is regarded as a dimensionality of<br /> ~ the target and domain units for the layer,<br /> ~ as `shape = (target, domain)`.<br /><br />Args:<br /> ~ shape (tuple): shape of the weight tensor (int, int)<br /><br />Returns:<br /> ~ weights layer


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


### derivatives
```py

def derivatives(self, units_target, units_domain, penalize=True)

```



Compute the derivative of the weights layer.<br /><br />dW_{ij} = - rac{1}{num_samples} * \sum_{k} v_{ki} h_{kj}<br /><br />Args:<br /> ~ units_target (tensor (num_samples, num_visible)): Rescaled target units.<br /> ~ units_domain (tensor (num_samples, num_visible)): Rescaled domain units.<br /><br />Returns:<br /> ~ derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)


### energy
```py

def energy(self, target_units, domain_units)

```



Compute the contribution of the weight layer to the model energy.<br /><br />For sample k:<br />E_k = -\sum_{ij} W_{ij} v_{ki} h_{kj}<br /><br />Args:<br /> ~ target_units (tensor (num_samples, num_visible)): Rescaled target units.<br /> ~ domain_units (tensor (num_samples, num_visible)): Rescaled domain units.<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### get\_config
```py

def get_config(self)

```



Get the configuration dictionary of the weights layer.<br /><br />Args:<br /> ~ None:<br /><br />Returns:<br /> ~ configuration (dict):


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


### random\_derivatives
```py

def random_derivatives(self)

```



Return an object like the derivatives that is filled with random floats.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### zero\_derivatives
```py

def zero_derivatives(self)

```



Return an object like the derivatives that is filled with zeros.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)




## functions

### namedtuple
```py

def namedtuple(typename, field_names, *, verbose=False, rename=False, module=None)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)


### weights\_from\_config
```py

def weights_from_config(config)

```



Construct a layer from a configuration.<br /><br />Args:<br /> ~ A dictionary configuration of the layer metadata.<br /><br />Returns:<br /> ~ An object which is a subclass of 'Weights'.

