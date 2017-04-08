# Documentation for Layers (layers.py)

## class ExponentialLayer
Layer with Exponential units (non-negative).
### \_\_init\_\_
```py

def __init__(self, num_units)

```



Create a layer with Exponential units.<br /><br />Args:<br /> ~ num_units (int): the size of the layer<br /><br />Returns:<br /> ~ exponential layer


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

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the intrinsic layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor, (num_units, num_connected_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, data)

```



Compute the energy of the Exponential layer.<br /><br />For sample k,<br />E_k = \sum_i loc_i * v_i<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the intrinsic parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### get\_base\_config
```py

def get_base_config(self)

```



Get a base configuration for the layer.<br /><br />Notes:<br /> ~ Encodes metadata for the layer.<br /> ~ Includes the base layer data.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the layer.


### get\_config
```py

def get_config(self)

```



Get the configuration dictionary of the Exponential layer.<br /><br />Args:<br /> ~ None:<br /><br />Returns:<br /> ~ configuratiom (dict):


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


### log\_partition\_function
```py

def log_partition_function(self, phi)

```



Compute the logarithm of the partition function of the layer<br />with external field phi.<br /><br />Let a_i be the intrinsic loc parameter of unit i.<br />Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.<br /><br />Z_i = Tr_{x_i} exp( -a_i x_i + phi_i x_i)<br />= 1 / (a_i - phi_i)<br /><br />log(Z_i) = -log(a_i - phi_i)<br /><br />Args:<br /> ~ phi (tensor (num_samples, num_units)): external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### mean
```py

def mean(self)

```



Compute the mean of the distribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mean of the distribution.


### mode
```py

def mode(self)

```



The mode of the Exponential distribution is undefined.<br /><br />Args:<br /> ~ None<br /><br />Raises:<br /> ~ NotImplementedError


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the intrinsic parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.int_params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the intrinsic parameters:<br /><br />layer.int_params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.int_params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For an Exponential layer, draws from the exponential distribution<br />with mean 1 (i.e., Expo(1)).<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Rescale is equivalent to the identity function for the Exponential layer.<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: observations


### sample\_state
```py

def sample_state(self)

```



Draw a random sample from the disribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): Sampled units.


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the intrinsic parameters of the layer.<br />Does nothing for the Exponential layer.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, scaled_units, weights, beta=None)

```



Update the extrinsic parameters of the layer.<br /><br />Notes:<br /> ~ Modfies layer.ext_params in place.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ None


## class ExtrinsicParams
ExtrinsicParams(rate,)


## class IntrinsicParams
IntrinsicParams(loc,)


## class Params
Params()




## class BernoulliLayer
Layer with Bernoulli units (i.e., 0 or +1).
### \_\_init\_\_
```py

def __init__(self, num_units)

```



Create a layer with Bernoulli units.<br /><br />Args:<br /> ~ num_units (int): the size of the layer<br /><br />Returns:<br /> ~ bernoulli layer


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

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the intrinsic layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor, (num_units, num_connected_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, data)

```



Compute the energy of the Bernoulli layer.<br /><br />For sample k,<br />E_k = -\sum_i loc_i * v_i<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the intrinsic parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### get\_base\_config
```py

def get_base_config(self)

```



Get a base configuration for the layer.<br /><br />Notes:<br /> ~ Encodes metadata for the layer.<br /> ~ Includes the base layer data.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the layer.


### get\_config
```py

def get_config(self)

```



Get the configuration dictionary of the Bernoulli layer.<br /><br />Args:<br /> ~ None:<br /><br />Returns:<br /> ~ configuratiom (dict):


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


### log\_partition\_function
```py

def log_partition_function(self, phi)

```



Compute the logarithm of the partition function of the layer<br />with external field phi.<br /><br />Let a_i be the intrinsic loc parameter of unit i.<br />Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.<br /><br />Z_i = Tr_{x_i} exp( a_i x_i + phi_i x_i)<br />= 1 + exp(a_i + phi_i)<br /><br />log(Z_i) = softplus(a_i + phi_i)<br /><br />Args:<br /> ~ phi (tensor (num_samples, num_units)): external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### mean
```py

def mean(self)

```



Compute the mean of the distribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mean of the distribution.


### mode
```py

def mode(self)

```



Compute the mode of the distribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mode of the distribution


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the intrinsic parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.int_params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the intrinsic parameters:<br /><br />layer.int_params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.int_params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For a Bernoulli layer, draws 0 or 1 with equal probability.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Rescale is equivalent to the identity function for the Bernoulli layer.<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: observations


### sample\_state
```py

def sample_state(self)

```



Draw a random sample from the disribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): Sampled units.


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the intrinsic parameters of the layer.<br />Does nothing for the Bernoulli layer.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, scaled_units, weights, beta=None)

```



Update the extrinsic parameters of the layer.<br /><br />Notes:<br /> ~ Modfies layer.ext_params in place.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ None


## class ExtrinsicParams
ExtrinsicParams(field,)


## class IntrinsicParams
IntrinsicParams(loc,)


## class Params
Params()




## class GaussianLayer
Layer with Gaussian units
### \_\_init\_\_
```py

def __init__(self, num_units)

```



Create a layer with Gaussian units.<br /><br />Args:<br /> ~ num_units (int): the size of the layer<br /><br />Returns:<br /> ~ gaussian layer


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

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the intrinsic layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor (num_units, num_connected_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, vis)

```



Compute the energy of the Gaussian layer.<br /><br />For sample k,<br />E_k = rac{1}{2} \sum_i rac{(v_i - loc_i)**2}{var_i}<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the intrinsic parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### get\_base\_config
```py

def get_base_config(self)

```



Get a base configuration for the layer.<br /><br />Notes:<br /> ~ Encodes metadata for the layer.<br /> ~ Includes the base layer data.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the layer.


### get\_config
```py

def get_config(self)

```



Get the configuration dictionary of the Gaussian layer.<br /><br />Args:<br /> ~ None:<br /><br />Returns:<br /> ~ configuration (dict):


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


### log\_partition\_function
```py

def log_partition_function(self, phi)

```



Compute the logarithm of the partition function of the layer<br />with external field phi.<br /><br />Let u_i and s_i be the intrinsic loc and scale parameters of unit i.<br />Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.<br /><br />Z_i = \int d x_i exp( -(x_i - u_i)^2 / (2 s_i^2) + \phi_i x_i)<br />= exp(b_i u_i + b_i^2 s_i^2 / 2) sqrt(2 pi) s_i<br /><br />log(Z_i) = log(s_i) + phi_i u_i + phi_i^2 s_i^2 / 2<br /><br />Args:<br /> ~ phi tensor (num_samples, num_units): external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### mean
```py

def mean(self)

```



Compute the mean of the distribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mean of the distribution.


### mode
```py

def mode(self)

```



Compute the mode of the distribution.<br />For a Gaussian layer, the mode equals the mean.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mode of the distribution


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the intrinsic parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.int_params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the intrinsic parameters:<br /><br />layer.int_params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.int_params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For a Gaussian layer, draws from the standard normal distribution N(0,1).<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Scale the observations by the variance of the layer.<br /><br />v'_i = v_i / var_i<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: Rescaled observations


### sample\_state
```py

def sample_state(self)

```



Draw a random sample from the disribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): Sampled units.


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=0.1)

```



Apply shrinkage to the variance parameters of the layer.<br /><br />new_variance = (1-shrinkage) * old_variance + shrinkage * 1<br /><br />Notes:<br /> ~ Modifies layer.int_params in place.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, scaled_units, weights, beta=None)

```



Update the extrinsic parameters of the layer.<br /><br />Notes:<br /> ~ Modfies layer.ext_params in place.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ None


## class ExtrinsicParams
ExtrinsicParams(mean, variance)


## class IntrinsicParams
IntrinsicParams(loc, log_var)


## class Params
Params()




## class OrderedDict
Dictionary that remembers insertion order


## class IsingLayer
Layer with Ising units (i.e., -1 or +1).
### \_\_init\_\_
```py

def __init__(self, num_units)

```



Create a layer with Ising units.<br /><br />Args:<br /> ~ num_units (int): the size of the layer<br /><br />Returns:<br /> ~ ising layer


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

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the intrinsic layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor, (num_units, num_connected_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, data)

```



Compute the energy of the Ising layer.<br /><br />For sample k,<br />E_k = -\sum_i loc_i * v_i<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the intrinsic parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### get\_base\_config
```py

def get_base_config(self)

```



Get a base configuration for the layer.<br /><br />Notes:<br /> ~ Encodes metadata for the layer.<br /> ~ Includes the base layer data.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the layer.


### get\_config
```py

def get_config(self)

```



Get the configuration dictionary of the Ising layer.<br /><br />Args:<br /> ~ None:<br /><br />Returns:<br /> ~ configuratiom (dict):


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


### log\_partition\_function
```py

def log_partition_function(self, phi)

```



Compute the logarithm of the partition function of the layer<br />with external field phi.<br /><br />Let a_i be the intrinsic loc parameter of unit i.<br />Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.<br /><br />Z_i = Tr_{x_i} exp( a_i x_i + phi_i x_i)<br />= 2 cosh(a_i + phi_i)<br /><br />log(Z_i) = logcosh(a_i + phi_i)<br /><br />Args:<br /> ~ phi (tensor (num_samples, num_units)): external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### mean
```py

def mean(self)

```



Compute the mean of the distribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mean of the distribution.


### mode
```py

def mode(self)

```



Compute the mode of the distribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mode of the distribution


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the intrinsic parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.int_params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the intrinsic parameters:<br /><br />layer.int_params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.int_params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For an Ising layer, draws -1 or +1 with equal probablity.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Rescale is equivalent to the identity function for the Ising layer.<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: observations


### sample\_state
```py

def sample_state(self)

```



Draw a random sample from the disribution.<br /><br />Determined from the extrinsic parameters (layer.ext_params).<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): Sampled units.


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the intrinsic parameters of the layer.<br />Does nothing for the Ising layer.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None


### update
```py

def update(self, scaled_units, weights, beta=None)

```



Update the extrinsic parameters of the layer.<br /><br />Notes:<br /> ~ Modfies layer.ext_params in place.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ None


## class ExtrinsicParams
ExtrinsicParams(field,)


## class IntrinsicParams
IntrinsicParams(loc,)


## class Params
Params()




## class Weights
Layer class for weights
### W
```py

def W(self)

```



Get the weight matrix.<br /><br />A convenience method for accessing layer.int_params.matrix<br />with a shorter syntax.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor: weight matrix


### W\_T
```py

def W_T(self)

```



Get the transpose of the weight matrix.<br /><br />A convenience method for accessing the transpose of<br />layer.int_params.matrix with a shorter syntax.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ tensor: transpose of weight matrix


### \_\_init\_\_
```py

def __init__(self, shape)

```



Create a weight layer.<br /><br />Notes:<br /> ~ Simple weight layers only have a single internal parameter matrix.<br /> ~ They have no external parameters because they do not depend<br /> ~ on the state of anything else.<br /><br /> ~ The shape is regarded as a dimensionality of<br /> ~ the visible and hidden units for the layer,<br /> ~ as `shape = (visible, hidden)`.<br /><br />Args:<br /> ~ shape (tuple): shape of the weight tensor (int, int)<br /><br />Returns:<br /> ~ weights layer


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

def derivatives(self, vis, hid)

```



Compute the derivative of the weights layer.<br /><br />dW_{ij} = - rac{1}{num_samples} * \sum_{k} v_{ki} h_{kj}<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_visible)): Rescaled visible units.<br /> ~ hid (tensor (num_samples, num_visible)): Rescaled hidden units.<br /><br />Returns:<br /> ~ derivs (namedtuple): 'matrix': tensor (contains gradient)


### energy
```py

def energy(self, vis, hid)

```



Compute the contribution of the weight layer to the model energy.<br /><br />For sample k:<br />E_k = -\sum_{ij} W_{ij} v_{ki} h_{kj}<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_visible)): Rescaled visible units.<br /> ~ hid (tensor (num_samples, num_visible)): Rescaled hidden units.<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the intrinsic parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### get\_base\_config
```py

def get_base_config(self)

```



Get a base configuration for the layer.<br /><br />Notes:<br /> ~ Encodes metadata for the layer.<br /> ~ Includes the base layer data.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ A dictionary configuration for the layer.


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


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the intrinsic parameters:<br /><br />layer.int_params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.int_params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


## class IntrinsicParams
IntrinsicParams(matrix,)


## class Params
Params()




## class Layer
A general layer class with common functionality.
### \_\_init\_\_
```py

def __init__(self, *args, **kwargs)

```



Basic layer initalization method.<br /><br />Args:<br /> ~ *args: any arguments<br /> ~ **kwargs: any keyword arguments<br /><br />Returns:<br /> ~ layer


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



Apply the contraints to the layer parameters.<br /><br />Note:<br /> ~ Modifies the intrinsic parameters of the layer in place.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ None


### from\_config
```py

def from_config(config)

```



Construct the layer from the base configuration.<br /><br />Args:<br /> ~ A dictionary configuration of the layer metadata.<br /><br />Returns:<br /> ~ An object which is a subclass of `Layer`.


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


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the intrinsic parameters:<br /><br />layer.int_params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.int_params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


## class Params
Params()




## functions

### get
```py

def get(key)

```



### namedtuple
```py

def namedtuple(typename, field_names, verbose=False, rename=False)

```



Returns a new subclass of tuple with named fields.<br /><br />>>> Point = namedtuple('Point', ['x', 'y'])<br />>>> Point.__doc__ ~  ~  ~  ~    # docstring for the new class<br />'Point(x, y)'<br />>>> p = Point(11, y=22) ~  ~  ~  # instantiate with positional args or keywords<br />>>> p[0] + p[1] ~  ~  ~  ~  ~  # indexable like a plain tuple<br />33<br />>>> x, y = p ~  ~  ~  ~  ~  ~ # unpack like a regular tuple<br />>>> x, y<br />(11, 22)<br />>>> p.x + p.y ~  ~  ~  ~  ~    # fields also accessible by name<br />33<br />>>> d = p._asdict() ~  ~  ~  ~  # convert to a dictionary<br />>>> d['x']<br />11<br />>>> Point(**d) ~  ~  ~  ~  ~   # convert from a dictionary<br />Point(x=11, y=22)<br />>>> p._replace(x=100) ~  ~  ~    # _replace() is like str.replace() but targets named fields<br />Point(x=100, y=22)

