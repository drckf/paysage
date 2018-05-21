# Documentation for Layer (layer.py)

## class CumulantsTAP
CumulantsTAP(mean, variance)<br />Note: the expectation thoughout the TAP codebase is that both mean and variance are tensors of shape (num_samples>1, num_units) or (num_units) in which num_samples is some sampling multiplicity used in the tap calculations, not the SGD batch size.


## class OrderedDict
Dictionary that remembers insertion order


## class ParamsLayer
Params()


## class Layer
A general layer class with common functionality.
### \_\_init\_\_
```py

def __init__(self, num_units, center=False, *args, **kwargs)

```



Basic layer initialization method.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_units (int): number of units in the layer<br />&nbsp;&nbsp;&nbsp;&nbsp;center (bool): whether to center the layer<br />&nbsp;&nbsp;&nbsp;&nbsp;*args: any arguments<br />&nbsp;&nbsp;&nbsp;&nbsp;**kwargs: any keyword arguments<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;layer


### add\_constraint
```py

def add_constraint(self, constraint)

```



Add a parameter constraint to the layer.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the layer.contraints attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;constraint (dict): {param_name: constraint (paysage.constraints)}<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### add\_penalty
```py

def add_penalty(self, penalty)

```



Add a penalty to the layer.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modfies the layer.penalties attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (dict): {param_name: penalty (paysage.penalties)}<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the parameters of the layer in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### get\_base\_config
```py

def get_base_config(self)

```



Get a base configuration for the layer.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Encodes metadata for the layer.<br />&nbsp;&nbsp;&nbsp;&nbsp;Includes the base layer data.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;A dictionary configuration for the layer.


### get\_center
```py

def get_center(self)

```



Get the vector used for centering:<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor ~ (num_units,)


### get\_config
```py

def get_config(self)

```



Get a full configuration for the layer.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Encodes metadata on the layer.<br />&nbsp;&nbsp;&nbsp;&nbsp;Weights are separately retrieved.<br />&nbsp;&nbsp;&nbsp;&nbsp;Builds the base configuration.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;A dictionary configuration for the layer.


### get\_fixed\_params
```py

def get_fixed_params(self)

```



Get the params that are not trainable.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;fixed_params (List[str]): a list of the fixed parameter names


### get\_param\_names
```py

def get_param_names(self)

```



Return the field names of the params attribute.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;field names (List[str])


### get\_params
```py

def get_params(self)

```



Get the value of the layer params attribute.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;params (list[namedtuple]): length=1 list


### get\_penalties
```py

def get_penalties(self)

```



Get the value of the penalties:<br /><br />E.g., L2 penalty = (1/2) * penalty * \sum_i parameter_i ** 2<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict (float): the values of the penalty functions


### get\_penalty\_grad
```py

def get_penalty_grad(self, deriv, param_name)

```



Get the gradient of the penalties on a parameter.<br /><br />E.g., L2 penalty gradient = penalty * parameter_i<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;deriv (tensor): derivative of the parameter<br />&nbsp;&nbsp;&nbsp;&nbsp;param_name: name of the parameter<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: derivative including penalty


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Performs an IO operation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (pandas.HDFStore): the readable stream for the params.<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the path for the layer params.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### num\_parameters
```py

def num_parameters(self)

```



### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;deltas (List[namedtuple]): List[param_name: tensor] (update)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.  Includes the moments for the layer.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Performs an IO operation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (pandas.HDFStore): the writeable stream for the params.<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the path for the layer params.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_fixed\_params
```py

def set_fixed_params(self, fixed_params)

```



Set the params that are not trainable.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the layer.fixed_params attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;fixed_params (List[str]): a list of the fixed parameter names<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_params
```py

def set_params(self, new_params)

```



Set the value of the layer params attribute.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies layer.params in place.<br />&nbsp;&nbsp;&nbsp;&nbsp;Note: expects a length=1 list<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;new_params (list[namedtuple])<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_moments
```py

def update_moments(self, units)

```



Set a reference mean and variance of the layer<br />&nbsp;&nbsp;&nbsp;&nbsp;(used for centering and sampling).<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies layer.reference_mean attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;units (tensor (batch_size, self.len)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None



