# Documentation for Layers (layers.py)

## class GradientMagnetizationBernoulli
This class represents a Bernoulli layer's contribution to the gradient vector<br />of the Gibbs free energy.<br />The underlying data is isomorphic to the MagnetizationBernoulli object.<br />It provides two layer-wise functions used in the TAP method for training RBMs
### \_\_init\_\_
```py

def __init__(self, exp)

```



Initialize self.  See help(type(self)) for accurate signature.


### expectation
```py

def expectation(self)

```



Returns the vector of expectations of unit values


### grad\_GFE\_update\_down
```py

def grad_GFE_update_down(self, mag_lower, mag, w, ww)

```



Computes a layerwise magnetization gradient update according to the gradient<br /> of the Gibbs Free energy.<br /><br />Args:<br /> ~ mag_lower (magnetization object): magnetization of the lower layer<br /> ~ mag (magnetization object): magnetization of the current layer<br /> ~ w (float tensor): weight matrix mapping down from this layer to the<br /> ~  ~  ~  ~  ~   lower layer<br /> ~ ww (float tensor): cached square of the weight matrix<br /><br />Returns:<br /> ~ None


### grad\_GFE\_update\_up
```py

def grad_GFE_update_up(self, mag, mag_upper, w, ww)

```



Computes a layerwise magnetization gradient update according to the gradient<br /> of the Gibbs Free energy.<br /><br />Args:<br /> ~ mag (magnetization object): magnetization of the current layer<br /> ~ mag_upper (magnetization object): magnetization of the upper layer<br /> ~ w (float tensor): weight matrix mapping down to this layer from the<br /> ~  ~  ~  ~  ~   upper layer<br /> ~ ww (float tensor): cached square of the weight matrix<br /><br />Returns:<br /> ~ None


### variance
```py

def variance(self)

```



Returns the variance of unit values. For a Bernoulli layer this<br />is determined by the expectation




## class MagnetizationBernoulli
This class holds the magnetization data of a Bernoulli layer.<br />Such data consists of a vector of expectation values for the layer's units,<br />MagnetizationBernoulli.expect, which are a float-valued in [0,1].<br />The class presents a getter for the expectation as well as a<br />function to compute the variance.
### \_\_init\_\_
```py

def __init__(self, exp)

```



Initialize self.  See help(type(self)) for accurate signature.


### expectation
```py

def expectation(self)

```



Returns the vector of expectations of unit values


### variance
```py

def variance(self)

```



Returns the variance of unit values. For a Bernoulli layer this<br />is determined by the expectation




## class ParamsExponential
ParamsExponential(loc,)


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

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, data)

```



Compute the energy of the Exponential layer.<br /><br />For sample k,<br />E_k = \sum_i loc_i * v_i<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


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


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the readable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### log\_partition\_function
```py

def log_partition_function(self, phi)

```



Compute the logarithm of the partition function of the layer<br />with external field phi.<br /><br />Let a_i be the loc parameter of unit i.<br />Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.<br /><br />Z_i = Tr_{x_i} exp( -a_i x_i + phi_i x_i)<br />= 1 / (a_i - phi_i)<br /><br />log(Z_i) = -log(a_i - phi_i)<br /><br />Args:<br /> ~ phi (tensor (num_samples, num_units)): external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For an Exponential layer, draws from the exponential distribution<br />with the rate determined by the params attribute.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Rescale is equivalent to the identity function for the Exponential layer.<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: observations


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the parameters of the layer.<br />Does nothing for the Exponential layer.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None




## class ParamsBernoulli
ParamsBernoulli(loc,)


## class BernoulliLayer
Layer with Bernoulli units (i.e., 0 or +1).
### GFE\_derivatives
```py

def GFE_derivatives(self, mag)

```



Gradient of the Gibbs free energy with respect to local field parameters<br /><br />Args:<br /> ~ mag (magnetization object): magnetization of the layer<br /><br />Returns:<br /> ~ gradient parameters (ParamsBernoulli): gradient w.r.t. local fields of GFE


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

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, data)

```



Compute the energy of the Bernoulli layer.<br /><br />For sample k,<br />E_k = -\sum_i loc_i * v_i<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


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


### get\_random\_magnetization
```py

def get_random_magnetization(self)

```



Create a layer magnetization with random expectations.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ BernoulliMagnetization


### get\_zero\_magnetization
```py

def get_zero_magnetization(self)

```



Create a layer magnetization with zero expectations.<br /><br />Args:<br /> ~ None<br /><br />Returns:<br /> ~ BernoulliMagnetization


### grad\_log\_partition\_function
```py

def grad_log_partition_function(self, B, A)

```



Compute the gradient of the logarithm of the partition function with respect to<br />its local field parameter with external field B and quadratic interaction A.<br /><br />(d_a_i)softplus(a_i + B_i - A_i) = expit(a_i + B_i - A_i)<br /><br />Note: This function returns the mean parameters over a minibatch of input fields<br /><br />Args:<br /> ~ A (tensor (num_samples, num_units)): external field<br /> ~ B (tensor (num_samples, num_units)): diagonal quadratic external field<br /><br />Returns:<br /> ~ (d_a_i) logZ (tensor (num_samples, num_units)): gradient of the log partition function


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the readable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### log\_partition\_function
```py

def log_partition_function(self, B, A)

```



Compute the logarithm of the partition function of the layer<br />with external field B augmented with a quadratic, diagonal interaction A.<br /><br />Let a_i be the loc parameter of unit i.<br />Let B_i be a local field<br />Let A_i be a diagonal quadratic interaction<br /><br />Z_i = Tr_{x_i} exp( a_i x_i + B_i x_i - A_i x_i^2)<br />= 1 + \exp(a_i + B_i - A_i)<br /><br />log(Z_i) = softplus(a_i + B_i - A_i)<br /><br />Args:<br /> ~ A (tensor (num_samples, num_units)): external field<br /> ~ B (tensor (num_samples, num_units)): diagonal quadratic external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For a Bernoulli layer, draws 0 or 1 with the field determined<br />by the params attribute.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Rescale is equivalent to the identity function for the Bernoulli layer.<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: observations


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the parameters of the layer.<br />Does nothing for the Bernoulli layer.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None




## class ParamsGaussian
ParamsGaussian(loc, log_var)


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


### conditional\_mean
```py

def conditional_mean(self, scaled_units, weights, beta=None)

```



Compute the mean of the distribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mean of the distribution.


### conditional\_mode
```py

def conditional_mode(self, scaled_units, weights, beta=None)

```



Compute the mode of the distribution conditioned on the state<br />of the connected layers. For a Gaussian layer, the mode equals<br />the mean.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): The mode of the distribution


### conditional\_sample
```py

def conditional_sample(self, scaled_units, weights, beta=None)

```



Draw a random sample from the disribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br /> ~ scaled_units list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the connected units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ tensor (num_samples, num_units): Sampled units.


### derivatives
```py

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, vis)

```



Compute the energy of the Gaussian layer.<br /><br />For sample k,<br />E_k = rac{1}{2} \sum_i rac{(v_i - loc_i)**2}{var_i}<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


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


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the readable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### log\_partition\_function
```py

def log_partition_function(self, phi)

```



Compute the logarithm of the partition function of the layer<br />with external field phi.<br /><br />Let u_i and s_i be the loc and scale parameters of unit i.<br />Let phi_i be an external field<br /><br />Z_i = \int d x_i exp( -(x_i - u_i)^2 / (2 s_i^2) + \phi_i x_i)<br />= exp(b_i u_i + b_i^2 s_i^2 / 2) sqrt(2 pi) s_i<br /><br />log(Z_i) = log(s_i) + phi_i u_i + phi_i^2 s_i^2 / 2<br /><br />Args:<br /> ~ phi tensor (num_samples, num_units): external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For a Gaussian layer, draws from a normal distribution<br />with the mean and variance determined from the params attribute.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Scale the observations by the variance of the layer.<br /><br />v'_i = v_i / var_i<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: Rescaled observations


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=0.1)

```



Apply shrinkage to the variance parameters of the layer.<br /><br />new_variance = (1-shrinkage) * old_variance + shrinkage * 1<br /><br />Notes:<br /> ~ Modifies layer.params in place.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None




## class ParamsWeights
ParamsWeights(matrix,)


## class OrderedDict
Dictionary that remembers insertion order


## class ParamsIsing
ParamsIsing(loc,)


## class ParamsLayer
Params()


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

def derivatives(self, vis, hid, weights, beta=None)

```



Compute the derivatives of the layer parameters.<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)):<br /> ~  ~ The values of the visible units.<br /> ~ hid list[tensor (num_samples, num_connected_units)]:<br /> ~  ~ The rescaled values of the hidden units.<br /> ~ weights list[tensor, (num_connected_units, num_units)]:<br /> ~  ~ The weights connecting the layers.<br /> ~ beta (tensor (num_samples, 1), optional):<br /> ~  ~ Inverse temperatures.<br /><br />Returns:<br /> ~ grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, data)

```



Compute the energy of the Ising layer.<br /><br />For sample k,<br />E_k = -\sum_i loc_i * v_i<br /><br />Args:<br /> ~ vis (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br /> ~ tensor (num_samples,): energy per sample


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


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the readable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### log\_partition\_function
```py

def log_partition_function(self, phi)

```



Compute the logarithm of the partition function of the layer<br />with external field phi.<br /><br />Let a_i be the loc parameter of unit i.<br />Let phi_i = \sum_j W_{ij} y_j, where y is the vector of connected units.<br /><br />Z_i = Tr_{x_i} exp( a_i x_i + phi_i x_i)<br />= 2 cosh(a_i + phi_i)<br /><br />log(Z_i) = logcosh(a_i + phi_i)<br /><br />Args:<br /> ~ phi (tensor (num_samples, num_units)): external field<br /><br />Returns:<br /> ~ logZ (tensor, num_samples, num_units)): log partition function


### online\_param\_update
```py

def online_param_update(self, data)

```



Update the parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br /> ~ Modifies layer.sample_size and layer.params in place.<br /><br />Args:<br /> ~ data (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For an Ising layer, draws -1 or +1 with the field determined<br />by the params attribute.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br /> ~ array_or_shape (array or shape tuple):<br /> ~  ~ If tuple, then this is taken to be the shape.<br /> ~  ~ If array, then its shape is used.<br /><br />Returns:<br /> ~ tensor: Random sample with desired shape.


### rescale
```py

def rescale(self, observations)

```



Rescale is equivalent to the identity function for the Ising layer.<br /><br />Args:<br /> ~ observations (tensor (num_samples, num_units)):<br /> ~  ~ Values of the observed units.<br /><br />Returns:<br /> ~ tensor: observations


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the parameters of the layer.<br />Does nothing for the Ising layer.<br /><br />Args:<br /> ~ shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br /> ~ None




## class Weights
Layer class for weights
### GFE\_derivatives
```py

def GFE_derivatives(self, vis, hid)

```



Gradient of the Gibbs free energy associated with this layer<br /><br />Args:<br /> ~ vis (magnetization object): magnetization of the lower layer linked to w<br /> ~ hid (magnetization objet): magnetization of the upper layer linked to w<br /><br />Returns:<br /> ~ derivs (namedtuple): 'matrix': tensor (contains gradient)


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



Create a weight layer.<br /><br />Notes:<br /> ~ The shape is regarded as a dimensionality of<br /> ~ the visible and hidden units for the layer,<br /> ~ as `shape = (visible, hidden)`.<br /><br />Args:<br /> ~ shape (tuple): shape of the weight tensor (int, int)<br /><br />Returns:<br /> ~ weights layer


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



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None




## class Layer
A general layer class with common functionality.
### \_\_init\_\_
```py

def __init__(self, *args, **kwargs)

```



Basic layer initialization method.<br /><br />Args:<br /> ~ *args: any arguments<br /> ~ **kwargs: any keyword arguments<br /><br />Returns:<br /> ~ layer


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


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the readable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br /> ~ Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br /> ~ deltas (dict): {param_name: tensor (update)}<br /><br />Returns:<br /> ~ None


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.<br /><br />Notes:<br /> ~ Performs an IO operation.<br /><br />Args:<br /> ~ store (pandas.HDFStore): the writeable stream for the params.<br /> ~ key (str): the path for the layer params.<br /><br />Returns:<br /> ~ None




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

