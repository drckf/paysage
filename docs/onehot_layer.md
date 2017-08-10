# Documentation for Onehot_Layer (onehot_layer.py)

## class CumulantsTAP
CumulantsTAP(mean, variance)


## class ParamsOneHot
ParamsOneHot(loc,)


## class OneHotLayer
Layer with 1-hot Bernoulli units.<br />e.g. for 3 units, the valid states are [1, 0, 0], [0, 1, 0], and [0, 0, 1].<br /><br />Dropout is unused.
### GFE\_derivatives
```py

def GFE_derivatives(self, cumulants, penalize=True)

```



Gradient of the Gibbs free energy with respect to local field parameters<br /><br />Args:<br /> ~ cumulants (CumulantsTAP object): magnetization of the layer<br /><br />Returns:<br /> ~ gradient parameters (ParamsOneHot): gradient w.r.t. local fields of GFE


### TAP\_entropy
```py

def TAP_entropy(self, lagrange, cumulants)

```



The TAP-0 Gibbs free energy term associated strictly with this layer<br /><br />Args:<br /> ~ lagrange (CumulantsTAP): Lagrange multiplers<br /> ~ cumulants (CumulantsTAP): magnetization of the layer<br /><br />Returns:<br /> ~ (float): 0th order term of Gibbs free energy


### TAP\_magnetization\_grad
```py

def TAP_magnetization_grad(self, mag, connected_mag, connected_weights)

```



Gradient of the Gibbs free energy with respect to the magnetization<br />associated strictly with this layer.<br /><br />Args:<br /> ~ mag (CumulantsTAP object): magnetization of the layer<br /> ~ connected_mag list[CumulantsTAP]: magnetizations of the connected layers<br /> ~ connected weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /><br />Return:<br /> ~ gradient of GFE w.r.t. magnetization (CumulantsTAP)


### \_\_init\_\_
```py

def __init__(self, num_units, dropout_p=0)

```



Create a layer with 1-hot units.<br /><br />Args:<br /> ~ num_units (int): the size of the layer<br /> ~ dropout_p (float): unused in this layer.<br /><br />Returns:<br /> ~ 1-hot layer


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


### clip\_magnetization
```py

def clip_magnetization(self, magnetization, a_min=1e-06, a_max=0.99999899)

```



Clip the mean of the mean of a CumulantsTAP object.<br /><br />Args:<br /> ~ magnetization (CumulantsTAP) to clip<br /> ~ a_min (float): the minimum value<br /> ~ a_max (float): the maximum value<br /><br />Returns:<br /> ~ clipped magnetization (CumulantsTAP)


### conditional\_mean
```py

def conditional_mean(self, scaled_units, weights, beta=None)

```



Compute the mean of the distribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mean of the distribution.


### conditional\_mode
```py

def conditional_mode(self, scaled_units, weights, beta=None)

```



Compute the mode of the distribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mode of the distribution


### conditional\_sample
```py

def conditional_sample(self, scaled_units, weights, beta=None)

```



Draw a random sample from the disribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): Sampled units.


### derivatives
```py

def derivatives(self, units, connected_units, connected_weights, penalize=True)

```



Compute the derivatives of the layer parameters.<br /><br />Args:<br /> ~ units (tensor (num_samples, num_units)):<br /> ~  ~ The values of the layer units.<br /> ~ connected_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ connected_weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, units)

```



Compute the energy of the 1-hot layer.<br /><br />For sample k,<br />E_k = -\sum_i loc_i * v_i<br /><br />Args:<br /> ~ units (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


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


### get\_magnetization
```py

def get_magnetization(self, mean)

```



Compute a CumulantsTAP object for the OneHotLayer.<br /><br />Args:<br /> ~ expect (tensor (num_units,)): expected values of the units<br /><br />returns:<br /> ~ CumulantsTAP


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


### get\_random\_magnetization
```py

def get_random_magnetization(self, epsilon=0.0049999999)

```



Create a layer magnetization with random expectations.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ CumulantsTAP


### get\_zero\_magnetization
```py

def get_zero_magnetization(self)

```



Create a layer magnetization with zero expectations.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ CumulantsTAP


### grad\_log\_partition\_function
```py

def grad_log_partition_function(self, external_field, quadratic_field)

```



Compute the gradient of the logarithm of the partition function with respect to<br />its local field parameter with external field (B) and quadratic field (A).<br /><br />Note: This function returns the mean parameters over a minibatch of input fields<br /><br />Args:<br /> ~ external_field (tensor (num_samples, num_units)): external field<br /> ~ quadratic_field (tensor (num_samples, num_units)): quadratic field<br /><br />Returns:<br /> ~ (d_a_i) logZ (tensor (num_samples, num_units)): gradient of the log partition function


### lagrange\_multiplers
```py

def lagrange_multiplers(self, cumulants)

```



The Lagrange multipliers associated with the first and second<br />cumulants of the units.<br /><br />Args:<br /> ~ cumulants (CumulantsTAP object): cumulants<br /><br />Returns:<br /> ~ lagrange multipliers (CumulantsTAP)


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the readable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### log\_partition\_function
```py

def log_partition_function(self, external_field, quadratic_field)

```



Compute the logarithm of the partition function of the layer<br />with external field (B) and quadratic field (A).<br /><br />Args:<br /> ~ external_field (tensor (num_samples, num_units)): external field<br /> ~ quadratic_field (tensor (num_samples, num_units)): quadratic field<br /><br />Returns:<br /> ~ logZ (tensor (num_samples, num_units)): log partition function


### onehot
```py

def onehot(self, n)

```



Generate an (n x n) tensor where each row has one unit with maximum<br />activation and all other units with minimum activation.<br /><br />Args:<br /> ~ n (int): the number of units<br /><br />Returns:<br /> ~ tensor (n, n)


### online\_param\_update
```py

def online_param_update(self, units)

```



Update the parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.params in place.<br /><br />Args:<br /> ~ units (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (List[namedtuple]): List[param_name: tensor] (update)<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For a 1-hot layer, draws units with the field determined<br />by the params attribute.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### random\_derivatives
```py

def random_derivatives(self)

```



Return an object like the derivatives that is filled with random floats.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)


### rescale
```py

def rescale(self, observations)

```



Rescale is trivial on the 1-hot layer.<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: observations


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


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the parameters of the layer.<br />Does nothing for the 1-hot layer.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None


### use\_dropout
```py

def use_dropout(self)

```



Indicate if the layer has dropout.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ true of false


### zero\_derivatives
```py

def zero_derivatives(self)

```



Return an object like the derivatives that is filled with zeros.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ derivs (List[namedtuple]): List['matrix': tensor] (contains gradient)




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

