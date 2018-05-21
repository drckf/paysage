# Documentation for Weights (weights.py)

## class ParamsWeights
ParamsWeights(matrix,)


## class OrderedDict
Dictionary that remembers insertion order


## class Weights
Layer class for weights.
### GFE\_derivatives
```py

def GFE_derivatives(self, rescaled_target_cumulants, rescaled_domain_cumulants)

```



Gradient of the Gibbs free energy associated with this layer<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_target_cumulants (CumulantsTAP): rescaled magnetization of<br />&nbsp;&nbsp;&nbsp;&nbsp; the shallower layer linked to w<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_domain_cumulants (CumulantsTAP): rescaled magnetization of<br />&nbsp;&nbsp;&nbsp;&nbsp; the deeper layer linked to w<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;derivs (namedtuple): 'matrix': tensor (contains gradient)


### W
```py

def W(self, trans=False)

```



Get the weight matrix.<br /><br />A convenience method for accessing layer.params.matrix<br />with a shorter syntax.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;trans (optional; bool): transpose the matrix if true<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: weight matrix


### \_\_init\_\_
```py

def __init__(self, shape)

```



Create a weight layer.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;The shape is regarded as a dimensionality of<br />&nbsp;&nbsp;&nbsp;&nbsp;the target and domain units for the layer,<br />&nbsp;&nbsp;&nbsp;&nbsp;as `shape = (target, domain)`.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shape (tuple): shape of the weight tensor (int, int)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;weights layer


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


### derivatives
```py

def derivatives(self, units_target, units_domain, penalize=True, weighting_function=<function do_nothing>)

```



Compute the derivative of the weights layer.<br /><br />dW_{ij} = - rac{1}{num_samples} * \sum_{k} v_{ki} h_{kj}<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;units_target (tensor (num_samples, num_visible)): Rescaled target units.<br />&nbsp;&nbsp;&nbsp;&nbsp;units_domain (tensor (num_samples, num_visible)): Rescaled domain units.<br />&nbsp;&nbsp;&nbsp;&nbsp;penalize (bool): whether to add a penalty term.<br />&nbsp;&nbsp;&nbsp;&nbsp;weighting_function (function): a weighting function to apply<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to units when computing the gradient.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)


### energy
```py

def energy(self, target_units, domain_units)

```



Compute the contribution of the weight layer to the model energy.<br /><br />For sample k:<br />E_k = -\sum_{ij} W_{ij} v_{ki} h_{kj}<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;target_units (tensor (num_samples, num_visible)): Rescaled target units.<br />&nbsp;&nbsp;&nbsp;&nbsp;domain_units (tensor (num_samples, num_visible)): Rescaled domain units.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the parameters of the layer in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### get\_config
```py

def get_config(self)

```



Get the configuration dictionary of the weights layer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;configuration (dict):


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


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;deltas (List[namedtuple]): List[param_name: tensor] (update)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### random\_derivatives
```py

def random_derivatives(self)

```



Return an object like the derivatives that is filled with random floats.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Performs an IO operation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (pandas.HDFStore): the writeable stream for the params.<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the path for the layer params.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_fixed\_params
```py

def set_fixed_params(self, fixed_params)

```



Set the params that are not trainable.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the layer.fixed_params attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;fixed_params (List[str]): a list of the fixed parameter names<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_params
```py

def set_params(self, new_params)

```



Set the value of the layer params attribute.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies layer.params in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;new_params (namedtuple)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### zero\_derivatives
```py

def zero_derivatives(self)

```



Return an object like the derivatives that is filled with zeros.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)




## functions

### weights\_from\_config
```py

def weights_from_config(config)

```



Construct a layer from a configuration.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;A dictionary configuration of the layer metadata.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;An object which is a subclass of 'Weights'.

