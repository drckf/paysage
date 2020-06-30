# Documentation for Dbm (dbm.py)

## class BoltzmannMachine
General model class.<br />(i.e., Restricted Boltzmann Machines).<br /><br />Example usage:<br />'''<br />vis = BernoulliLayer(nvis)<br />hid = BernoulliLayer(nhid)<br />rbm = BoltzmannMachine([vis, hid])<br />'''
### TAP\_gradient
```py

def TAP_gradient(self, data_state, use_GD=True, init_lr=0.1, tol=1e-07, max_iters=50, ratchet=True, decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999)

```



Gradient of -\ln P(v) with respect to the model parameters<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;data_state (State object): The observed visible units and sampled<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;hidden units.<br />&nbsp;&nbsp;&nbsp;&nbsp;use_GD (bool): use gradient descent or use self_consistent iteration<br />&nbsp;&nbsp;&nbsp;&nbsp;init_lr (float): initial learning rate for GD<br />&nbsp;&nbsp;&nbsp;&nbsp;tol (float): tolerance for quitting minimization<br />&nbsp;&nbsp;&nbsp;&nbsp;max_iters (int): maximum gradient decsent steps<br />&nbsp;&nbsp;&nbsp;&nbsp;ratchet (bool): don't perform gradient update if not lowering GFE<br />&nbsp;&nbsp;&nbsp;&nbsp;decrease_on_neg (float): factor to multiply lr by if the gradient step<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fails to lower the GFE<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_weight (float): mean weight parameter for ADAM<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_square_weight (float): mean square weight parameter for ADAM<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;setting to 0.0 turns off adaptive weighting<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient (Gradient): containing gradients of the model parameters.


### \_\_init\_\_
```py

def __init__(self, layer_list: List, conn_list: List = None)

```



Create a model.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;layer_list (List[layer])<br />&nbsp;&nbsp;&nbsp;&nbsp;conn_list (optional; List[Connection])<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;BoltzmannMachine


### compute\_StateTAP
```py

def compute_StateTAP(self, use_GD=True, init_lr=0.1, tol=1e-07, max_iters=50, ratchet=True, decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999, seed=None, rescaled_weight_cache=None)

```



Compute the state of the layers by minimizing the second order TAP<br />approximation to the Helmholtz free energy.  This function selects one of two<br />possible implementations of this minimization procedure, gradient-descent or<br />self-consistent iteration.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;use_GD (bool): use gradient descent or use self_consistent iteration<br />&nbsp;&nbsp;&nbsp;&nbsp;init_lr (float): initial learning rate for GD<br />&nbsp;&nbsp;&nbsp;&nbsp;tol (float): tolerance for quitting minimization<br />&nbsp;&nbsp;&nbsp;&nbsp;max_iters (int): maximum gradient decsent steps<br />&nbsp;&nbsp;&nbsp;&nbsp;ratchet (bool): don't perform gradient update if not lowering GFE<br />&nbsp;&nbsp;&nbsp;&nbsp;decrease_on_neg (float): factor to multiply lr by if the gradient step<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fails to lower the GFE<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_weight (float): mean weight parameter for ADAM<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_square_weight (float): mean square weight parameter for ADAM<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;setting to 0.0 turns off adaptive weighting<br />&nbsp;&nbsp;&nbsp;&nbsp;seed (CumulantsTAP): seed for the minimization<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_weight_cache tuple(list[tensor],list[tensor]): cache of<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rescaled weight and weight_square matrices<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tuple (StateTAP, float): TAP state of the layers and the GFE


### compute\_reconstructions
```py

def compute_reconstructions(self, visible, method='markov_chain')

```



Compute the reconstructions of a visible tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;visible (tensor (num_samples, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;method (str): ['markov_chain', 'mean_field_iteration', 'deterministic_iteration']<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;reconstructions (tensor (num_samples, num_units))


### copy
```py

def copy(self)

```



Copy a Boltzmann machine.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;BoltzmannMachine


### copy\_params
```py

def copy_params(self, model)

```



Copy the params from a source model into self.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies attributes in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model (BoltzmannMachine)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### count\_connections
```py

def count_connections(self)

```



Set the num_connections attribute.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the num_connections attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### deterministic\_iteration
```py

def deterministic_iteration(self, n: int, state: paysage.models.state.State, beta=None, callbacks=None) -> paysage.models.state.State

```



Perform multiple deterministic (maximum probability) updates<br />in alternating layers.<br />state -> new state<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Returns the layer units that maximize the probability<br />&nbsp;&nbsp;&nbsp;&nbsp;conditioned on adjacent layers,<br />&nbsp;&nbsp;&nbsp;&nbsp;x_i = argmax P(x_i | x_(i-1), x_(i+1))<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n (int): number of steps.<br />&nbsp;&nbsp;&nbsp;&nbsp;state (State object): the state of each layer<br />&nbsp;&nbsp;&nbsp;&nbsp;beta (optional, tensor (batch_size, 1)): Inverse temperatures<br />&nbsp;&nbsp;&nbsp;&nbsp;callbacks (optional, List[callable]): list of functions to call<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;at each step; signature func(State)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;new state


### exclusive\_gradient\_
```py

def exclusive_gradient_(self, grad, state, func, penalize=True, weighting_function=<function do_nothing>)

```



Compute the gradient of the model parameters using only a single phase.<br />Scales the units in the state and computes the gradient.<br />Includes a weight factor for the gradients.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies grad in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;grad (Gradient): a gradient object<br />&nbsp;&nbsp;&nbsp;&nbsp;state (State object): the state of the units<br />&nbsp;&nbsp;&nbsp;&nbsp;func (Callable): a function like func(tensor, tensor) -> tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;penalize (bool): control on applying layer penalties<br />&nbsp;&nbsp;&nbsp;&nbsp;weighting_function (function): a weighting function to apply<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to units when computing the gradient.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict: Gradients of the model parameters.


### get\_config
```py

def get_config(self) -> dict

```



Get a configuration for the model.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Includes metadata on the layers.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;A dictionary configuration for the model.


### get\_sampled
```py

def get_sampled(self)

```



Convenience function that returns the layers for which sampling is<br />not clamped.<br />Complement of the `clamped_sampling` attribute.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;unclamped_sampling (List): layers for which sampling is not clamped.


### gibbs\_free\_energy
```py

def gibbs_free_energy(self, cumulants, rescaled_weight_cache=None)

```



Gibbs Free Energy (GFE) according to TAP2 appoximation<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants list(CumulantsTAP): cumulants of the layers<br />&nbsp;&nbsp;&nbsp;&nbsp;rescaled_weight_cache tuple(list[tensor], list[tensor]):<br />&nbsp;&nbsp;&nbsp;&nbsp; cached list of rescaled weight matrices and squares thereof<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float: Gibbs free energy


### grad\_TAP\_free\_energy
```py

def grad_TAP_free_energy(self, use_GD=True, init_lr=0.1, tol=1e-07, max_iters=50, ratchet=True, decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999)

```



Compute the gradient of the Helmholtz free engergy of the model according<br />to the TAP expansion around infinite temperature.<br /><br />This function will use the class members which specify the parameters for<br />the Gibbs FE minimization.<br />The gradients are taken as the average over the gradients computed at<br />each of the minimial magnetizations for the Gibbs FE.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;use_GD (bool): use gradient descent or use self_consistent iteration<br />&nbsp;&nbsp;&nbsp;&nbsp;init_lr (float): initial learning rate for GD<br />&nbsp;&nbsp;&nbsp;&nbsp;tol (float): tolerance for quitting minimization<br />&nbsp;&nbsp;&nbsp;&nbsp;max_iters (int): maximum gradient decsent steps<br />&nbsp;&nbsp;&nbsp;&nbsp;ratchet (bool): don't perform gradient update if not lowering GFE<br />&nbsp;&nbsp;&nbsp;&nbsp;decrease_on_neg (float): factor to multiply lr by if the gradient step<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fails to lower the GFE<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_weight (float): mean weight parameter for ADAM<br />&nbsp;&nbsp;&nbsp;&nbsp;mean_square_weight (float): mean square weight parameter for ADAM<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;setting to 0.0 turns off adaptive weighting<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;namedtuple: (Gradient): containing gradients of the model parameters.


### gradient
```py

def gradient(self, data_state, model_state, data_weighting_function=<function do_nothing>, model_weighting_function=<function do_nothing>)

```



Compute the gradient of the model parameters.<br />Scales the units in the state and computes the gradient.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;data_state (State object): The observed visible units and<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sampled hidden units.<br />&nbsp;&nbsp;&nbsp;&nbsp;model_state (State object): The visible and hidden units<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sampled from the model.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict: Gradients of the model parameters.


### initialize
```py

def initialize(self, batch, method: str = 'hinton', **kwargs) -> None

```



Initialize the parameters of the model.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch: A Batch object.<br />&nbsp;&nbsp;&nbsp;&nbsp;method (optional): The initialization method.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### joint\_energy
```py

def joint_energy(self, state)

```



Compute the joint energy of the model based on a state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;state (State object): the current state of each layer<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples,): Joint energies.


### lagrange\_multipliers\_analytic
```py

def lagrange_multipliers_analytic(self, cumulants)

```



Compute lagrange multipliers of each layer according to an analytic calculation<br />&nbsp;&nbsp;&nbsp;&nbsp;of lagrange multipliers at beta=0.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;cumulants (list[CumulantsTAP]): list of magnetizations<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of each layer<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;lagrange_multipliers (list [CumulantsTAP])


### markov\_chain
```py

def markov_chain(self, n: int, state: paysage.models.state.State, beta=None, callbacks=None) -> paysage.models.state.State

```



Perform multiple Gibbs sampling steps in alternating layers.<br />state -> new state<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Samples layers according to the conditional probability<br />&nbsp;&nbsp;&nbsp;&nbsp;on adjacent layers,<br />&nbsp;&nbsp;&nbsp;&nbsp;x_i ~ P(x_i | x_(i-1), x_(i+1) )<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n (int): number of steps.<br />&nbsp;&nbsp;&nbsp;&nbsp;state (State object): the state of each layer<br />&nbsp;&nbsp;&nbsp;&nbsp;beta (optional, tensor (batch_size, 1)): Inverse temperatures<br />&nbsp;&nbsp;&nbsp;&nbsp;callbacks(optional, List[callable]): list of functions to call<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;at each step; signature func(State)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;new state


### mean\_field\_iteration
```py

def mean_field_iteration(self, n: int, state: paysage.models.state.State, beta=None, callbacks=None) -> paysage.models.state.State

```



Perform multiple mean-field updates in alternating layers<br />state -> new state<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Returns the expectation of layer units<br />&nbsp;&nbsp;&nbsp;&nbsp;conditioned on adjacent layers,<br />&nbsp;&nbsp;&nbsp;&nbsp;x_i = E[x_i | x_(i-1), x_(i+1) ]<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n (int): number of steps.<br />&nbsp;&nbsp;&nbsp;&nbsp;state (State object): the state of each layer<br />&nbsp;&nbsp;&nbsp;&nbsp;beta (optional, tensor (batch_size, 1)): Inverse temperatures<br />&nbsp;&nbsp;&nbsp;&nbsp;callbacks (optional, List[callable]): list of functions to call<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;at each step; signature func(State)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;new state


### num\_parameters
```py

def num_parameters(self)

```



Return the number of parameters in the model<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;number of parameters


### parameter\_update
```py

def parameter_update(self, deltas)

```



Update the model parameters.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the model parameters in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;deltas (Gradient)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### save
```py

def save(self, store: pandas.io.pytables.HDFStore) -> None

```



Save a model to an open HDFStore.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Performs an IO operation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (pandas.HDFStore)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_clamped\_sampling
```py

def set_clamped_sampling(self, clamped_sampling)

```



Convenience function to set the layers for which sampling is clamped.<br />Sets exactly the given layers to have sampling clamped.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;clamped_sampling (List): the exact set of layers which are have sampling clamped.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class partial
partial(func, *args, **keywords) - new function with partial application<br />of the given arguments and keywords.

