# Documentation for Samplers (samplers.py)

## class AutoregressiveGammaSampler
Sampler from an autoregressive Gamma process.
### \_\_init\_\_
```py

def __init__(self, beta_momentum=0.9, beta_std=0.6, schedule=<paysage.schedules.Constant object>)

```



Create an autoregressive gamma sampler.<br />Can be used to sample inverse temperatures for MC sampling.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_momentum (float in [0,1]; optional): autoregressive coefficient<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the inverse temperature, beta.<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_std (float >= 0; optional): the standard deviation of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;inverse temperature, beta.<br />&nbsp;&nbsp;&nbsp;&nbsp;schedule (generator; optional)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;An AutoregressiveGammaSampler instance.


### get\_beta
```py

def get_beta(self)

```



Return beta in the appropriate tensor format.


### set\_schedule
```py

def set_schedule(self, value)

```



Change the value of the learning rate schedule.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the schedule.value attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;value (float)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_std
```py

def set_std(self, std, momentum=0.9)

```



Set the parameters based off the standard deviation.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies many layer attributes in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;std (float)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update\_beta
```py

def update_beta(self, num_samples)

```



Update beta with an autoregressive Gamma process.<br /><br />beta_0 ~ Gamma(nu,c/(1-phi)) = Gamma(nu, var)<br />h_t ~ Possion( phi/c * h_{t-1})<br />beta_t ~ Gamma(nu + z_t, c)<br /><br />Achieves a stationary distribution with mean 1 and variance var:<br />Gamma(nu, var) = Gamma(1/var, var)<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the folling attributes in place:<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;has_beta, beta_shape, beta<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_samples (int): the number of samples to generate for beta<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class SequentialMC
An accelerated sequential Monte Carlo sampler
### \_\_init\_\_
```py

def __init__(self, model, mcsteps=1, clamped=None, updater='markov_chain', beta_momentum=0.9, beta_std=0.6, schedule=<paysage.schedules.Constant object>)

```



Create a sequential Monte Carlo sampler.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model (BoltzmannMachine)<br />&nbsp;&nbsp;&nbsp;&nbsp;mcsteps (int; optional): the number of Monte Carlo steps<br />&nbsp;&nbsp;&nbsp;&nbsp;clamped (List[int]; optional): list of layers to clamp<br />&nbsp;&nbsp;&nbsp;&nbsp;updater (str; optional): method for updating the state<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_momentum (float in [0,1]; optional): autoregressive coefficient<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the inverse temperature of beta<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_std (float >= 0; optional): the standard deviation of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;inverse temperature beta<br />&nbsp;&nbsp;&nbsp;&nbsp;schedule (generator; optional)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;SequentialMC


### reset
```py

def reset(self)

```



Reset the sampler state.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies sampler.state attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_state
```py

def set_state(self, state)

```



Set the state.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;state (State): The state of the units.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_state\_from\_visible
```py

def set_state_from_visible(self, vdata)

```



Set the state of the sampler using a sample of visible vectors.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the sampler.state attribute in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vdata (tensor~(num_samples,num_units)): a visible state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### state\_for\_grad
```py

def state_for_grad(self, target_layer)

```



Peform a mean field update of the target layer.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;target_layer (int): the layer to update<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;state


### update\_state
```py

def update_state(self, steps=None)

```



Update the state of the particles.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state attribute in place.<br />&nbsp;&nbsp;&nbsp;&nbsp;Calls the beta_sampler.update_beta() method.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;steps (int): the number of Monte Carlo steps<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None



