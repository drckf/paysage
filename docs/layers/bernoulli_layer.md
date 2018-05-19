# Documentation for Bernoulli_Layer (bernoulli_layer.py)

## class ParamsBernoulli
ParamsBernoulli(loc,)


## class BernoulliLayer
Layer with Bernoulli units (i.e., 0 or +1).
### GFE\_derivatives
```py

def GFE_derivatives(self, cumulants, connected_cumulants=None, rescaled_connected_weights=None, rescaled_connected_weights_sq=None)

```



Gradient of the Gibbs free energy with respect to local field parameters<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (CumulantsTAP object): magnetization of the layer<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient parameters (ParamsBernoulli): gradient w.r.t. local fields of GFE


### TAP\_entropy
```py

def TAP_entropy(self, cumulants)

```



The TAP-0 Gibbs free energy term associated strictly with this layer<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (CumulantsTAP): magnetization of the layer<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;(float): 0th order term of Gibbs free energy


### TAP\_magnetization\_grad
```py

def TAP_magnetization_grad(self, cumulants, connected_cumulants, rescaled_connected_weights, rescaled_connected_weights_sq)

```



Gradient of the Gibbs free energy with respect to the magnetization<br />associated strictly with this layer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (CumulantsTAP): magnetization of the layer<br />&nbsp;&nbsp;&nbsp;&nbsp;connected_cumulants (list[CumulantsTAP]): magnetizations of the connected layers<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_connected_weights (list[tensor, (num_connected_units, num_units)]):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_connected_weights_sq (list[tensor, (num_connected_units, num_units)]):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The cached squares of weights connecting the layers.<br /><br />Return:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient of GFE w.r.t. magnetization (CumulantsTAP)


### \_\_init\_\_
```py

def __init__(self, num_units, center=False)

```



Create a layer with Bernoulli units.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_units (int): the size of the layer<br />&nbsp;&nbsp;&nbsp;&nbsp;center (bool): whether to center the layer<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Bernoulli layer


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


### clip\_magnetization
```py

def clip_magnetization(self, magnetization, a_min=1e-06, a_max=0.999999)

```



Clip the mean of the mean of a CumulantsTAP object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;magnetization (CumulantsTAP) to clip<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (float): the minimum value<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (float): the maximum value<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;clipped magnetization (CumulantsTAP)


### clip\_magnetization\_
```py

def clip_magnetization_(self, magnetization, a_min=1e-06, a_max=0.999999)

```



Clip the mean of the mean of a CumulantsTAP object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;magnetization (CumulantsTAP) to clip<br />&nbsp;&nbsp;&nbsp;&nbsp;a_min (float): the minimum value<br />&nbsp;&nbsp;&nbsp;&nbsp;a_max (float): the maximum value<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### conditional\_mean
```py

def conditional_mean(self, scaled_units, weights, beta=None)

```



Compute the mean of the distribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;scaled_units list[tensor (num_samples, num_connected_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The rescaled values of the connected units.<br />&nbsp;&nbsp;&nbsp;&nbsp;weights list[tensor (num_connected_units, num_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;beta (tensor (num_samples, 1), optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Inverse temperatures.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units): The mean of the distribution.


### conditional\_mode
```py

def conditional_mode(self, scaled_units, weights, beta=None)

```



Compute the mode of the distribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;scaled_units list[tensor (num_samples, num_connected_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The rescaled values of the connected units.<br />&nbsp;&nbsp;&nbsp;&nbsp;weights list[tensor (num_connected_units, num_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;beta (tensor (num_samples, 1), optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Inverse temperatures.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units): The mode of the distribution


### conditional\_params
```py

def conditional_params(self, scaled_units, weights, beta=None)

```



Compute the parameters of the layer conditioned on the state<br />of the connected layers.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;scaled_units list[tensor (num_samples, num_connected_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The rescaled values of the connected units.<br />&nbsp;&nbsp;&nbsp;&nbsp;weights list[tensor, (num_connected_units, num_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;beta (tensor (num_samples, 1), optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Inverse temperatures.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: conditional parameters


### conditional\_sample
```py

def conditional_sample(self, scaled_units, weights, beta=None)

```



Draw a random sample from the disribution conditioned on the state<br />of the connected layers.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;scaled_units list[tensor (num_samples, num_connected_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The rescaled values of the connected units.<br />&nbsp;&nbsp;&nbsp;&nbsp;weights list[tensor (num_connected_units, num_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;beta (tensor (num_samples, 1), optional):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Inverse temperatures.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units): Sampled units.


### derivatives
```py

def derivatives(self, units, connected_units, connected_weights, penalize=True, weighting_function=<function do_nothing>)

```



Compute the derivatives of the layer parameters.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;units (tensor (num_samples, num_units)):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The values of the layer units.<br />&nbsp;&nbsp;&nbsp;&nbsp;connected_units list[tensor (num_samples, num_connected_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The rescaled values of the connected units.<br />&nbsp;&nbsp;&nbsp;&nbsp;connected_weights list[tensor, (num_connected_units, num_units)]:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;penalize (bool): whether to add a penalty term.<br />&nbsp;&nbsp;&nbsp;&nbsp;weighting_function (function): a weighting function to apply<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to units when computing the gradient.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (namedtuple): param_name: tensor (contains gradient)


### energy
```py

def energy(self, units)

```



Compute the energy of the Bernoulli layer.<br /><br />For sample k,<br />E_k = -\sum_i loc_i * v_i<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;units (tensor (num_samples, num_units)): values of units<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples,): energy per sample


### enforce\_constraints
```py

def enforce_constraints(self)

```



Apply the contraints to the layer parameters.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the parameters of the layer in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### envelope\_random
```py

def envelope_random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For a Bernoulli layer, draws 0 or 1 from a bernoulli layer with mean<br />self.moments.mean<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;array_or_shape (array or shape tuple):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If tuple, then this is taken to be the shape.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If array, then its shape is used.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random sample with desired shape.


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


### get\_magnetization
```py

def get_magnetization(self, mean)

```



Compute a CumulantsTAP object for the BernoulliLayer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mean (tensor (num_units,)): expected values of the units<br /><br />returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;CumulantsTAP


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


### get\_random\_magnetization
```py

def get_random_magnetization(self, num_samples=1, epsilon=1e-06)

```



Create a layer magnetization with random expectations.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_samples (int>0): number of random samples to draw<br />&nbsp;&nbsp;&nbsp;&nbsp;epsilon (float): bound away from [0,1] in which to draw magnetization values<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;CumulantsTAP


### get\_zero\_magnetization
```py

def get_zero_magnetization(self)

```



Create a layer magnetization with zero expectations.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;CumulantsTAP


### lagrange\_multipliers\_analytic
```py

def lagrange_multipliers_analytic(self, cumulants)

```



Return the Lagrange multipliers (at beta=0) according to the starionarity<br />&nbsp;&nbsp;&nbsp;&nbsp;conditions {d/da(GibbsFE)=0, d/dc(GibbsFE)=0} at beta=0.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (CumulantsTAP object): layer magnetization cumulants<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;lagrange multipliers (CumulantsTAP)


### load\_params
```py

def load_params(self, store, key)

```



Load the parameters from an HDFStore.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Performs an IO operation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (pandas.HDFStore): the readable stream for the params.<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the path for the layer params.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### log\_partition\_function
```py

def log_partition_function(self, external_field, quadratic_field)

```



Compute the logarithm of the partition function of the layer<br />with external field (B) and quadratic field (A).<br /><br />Let a_i be the loc parameter of unit i.<br />Let B_i be an external field<br />Let A_i be a quadratic field<br /><br />Z_i = Tr_{x_i} exp( a_i x_i + B_i x_i + A_i x_i^2)<br />= 1 + \exp(a_i + B_i + A_i)<br /><br />log(Z_i) = softplus(a_i + B_i + A_i)<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;external_field (tensor (num_samples, num_units)): external field<br />&nbsp;&nbsp;&nbsp;&nbsp;quadratic_field (tensor (num_samples, num_units)): quadratic field<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;logZ (tensor (num_samples, num_units)): log partition function


### num\_parameters
```py

def num_parameters(self)

```



### online\_param\_update
```py

def online_param_update(self, units)

```



Update the parameters using an observed batch of data.<br />Used for initializing the layer parameters.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies layer.params in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;units (tensor (num_samples, num_units)): observed values for units<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### parameter\_step
```py

def parameter_step(self, deltas)

```



Update the values of the parameters:<br /><br />layer.params.name -= deltas.name<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the elements of the layer.params attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;deltas (List[namedtuple]): List[param_name: tensor] (update)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### random
```py

def random(self, array_or_shape)

```



Generate a random sample with the same type as the layer.<br />For a Bernoulli layer, draws 0 or 1 with the field determined<br />by the params attribute.<br /><br />Used for generating initial configurations for Monte Carlo runs.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;array_or_shape (array or shape tuple):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If tuple, then this is taken to be the shape.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If array, then its shape is used.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: Random sample with desired shape.


### random\_derivatives
```py

def random_derivatives(self)

```



Return an object like the derivatives that is filled with random floats.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;derivs (List[namedtuple]): List[param_name: tensor] (contains gradient)


### reciprocal\_scale
```py

def reciprocal_scale(self)

```



Returns a tensor of shape (num_units) providing a reciprocal scale for each unit<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;reciproical scale (tensor)


### rescale
```py

def rescale(self, observations)

```



Rescale is trivial for the Bernoulli layer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;observations (tensor (num_samples, num_units)):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Values of the observed units.<br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: observations


### rescale\_cumulants
```py

def rescale_cumulants(self, cumulants)

```



Rescales the cumulants associated with the layer.<br /> Trivial for the Bernoulli layer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (CumulantsTAP)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled cumulants (CumulantsTAP)


### save\_params
```py

def save_params(self, store, key)

```



Save the parameters to a HDFStore.  Includes the moments for the layer.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Performs an IO operation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (pandas.HDFStore): the writeable stream for the params.<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the path for the layer params.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### self\_consistent\_update\_
```py

def self_consistent_update_(self, cumulants, lagrange_multipliers)

```



Applies self-consistent TAP update to the layer's magnetization. This formula<br /> is analytically computed --not based on a 2-term truncation of the Gibbs FE.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (CumulantsTAP object): magnetization of the layer<br />&nbsp;&nbsp;&nbsp;&nbsp;lagrange_multipliers (CumulantsTAP object)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


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


### shrink\_parameters
```py

def shrink_parameters(self, shrinkage=1)

```



Apply shrinkage to the parameters of the layer.<br />Does nothing for the Bernoulli layer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;shrinkage (float \in [0,1]): the amount of shrinkage to apply<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_lagrange\_multipliers\_
```py

def update_lagrange_multipliers_(self, cumulants, lagrange_multipliers, connected_cumulants, rescaled_connected_weights, rescaled_connected_weights_sq)

```



Update, in-place, the Lagrange multipliers with respect to the TAP2 approximation<br />of the GFE as in<br /><br />Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,<br />and Florent Krzakala<br />"A Deterministic and Generalized Framework for Unsupervised Learning<br />with Restricted Boltzmann Machines"<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (CumulantsTAP): layer magnetization cumulants<br />&nbsp;&nbsp;&nbsp;&nbsp;lagrange_multipliers (CumulantsTAP)<br />&nbsp;&nbsp;&nbsp;&nbsp;connected_cumulants (CumulantsTAP): connected magnetization cumulants<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_connected_weights (list[tensor, (num_connected_units, num_units)]):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_connected_weights_sq (list[tensor, (num_connected_units, num_units)]):<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The cached squares of weights connecting the layers.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(unused on Bernoulli layer)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_moments
```py

def update_moments(self, units)

```



Set a reference mean and variance of the layer<br />&nbsp;&nbsp;&nbsp;&nbsp;(used for centering and sampling).<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies layer.reference_mean attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;units (tensor (batch_size, self.len)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### zero\_derivatives
```py

def zero_derivatives(self)

```



Return an object like the derivatives that is filled with zeros.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;derivs (List[namedtuple]): List[param_name: tensor] (contains gradient)




## class CumulantsTAP
CumulantsTAP(mean, variance)<br />Note: the expectation thoughout the TAP codebase is that both mean and variance are tensors of shape (num_samples>1, num_units) or (num_units) in which num_samples is some sampling multiplicity used in the tap calculations, not the SGD batch size.


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



