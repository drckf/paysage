# Documentation for Sgd (sgd.py)

## class StochasticGradientDescent
Stochastic gradient descent with minibatches
### \_\_init\_\_
```py

def __init__(self, model, batch, fantasy_steps=10)

```



Create a StochasticGradientDescent object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a model object<br />&nbsp;&nbsp;&nbsp;&nbsp;batch: a batch object<br />&nbsp;&nbsp;&nbsp;&nbsp;fantasy_steps (int): the number of steps for fantasy particles<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in the progress monitor.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;StochasticGradientDescent


### train
```py

def train(self, optimizer, num_epochs, mcsteps=1, update_method='markov_chain', method=<function persistent_contrastive_divergence>, beta_std=0.6, negative_phase_batch_size=None, verbose=True, burn_in=0)

```



Train the model.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Updates the model parameters in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;optimizer: an optimizer object<br />&nbsp;&nbsp;&nbsp;&nbsp;num_epochs (int): the number of epochs<br />&nbsp;&nbsp;&nbsp;&nbsp;mcsteps (int; optional): the number of Monte Carlo steps per gradient<br />&nbsp;&nbsp;&nbsp;&nbsp;update_method (str; optional): the method used to update the state<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[markov_chain, deterministic_iteration, mean_field_iteration]<br />&nbsp;&nbsp;&nbsp;&nbsp;method (fit.methods obj; optional): the method used to approximate the likelihood<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   gradient [cd, pcd, tap]<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_std (float; optional): the standard deviation of the inverse<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;temperature of the SequentialMC sampler<br />&nbsp;&nbsp;&nbsp;&nbsp;negative_phase_batch_size (int; optional): the batch size for the negative phase.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If None, matches the positive_phase batch size.<br />&nbsp;&nbsp;&nbsp;&nbsp;verbose (bool; optional): print output to stdout<br />&nbsp;&nbsp;&nbsp;&nbsp;burn_in (int; optional): the number of initial epochs during which<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the beta_std will be set to 0<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None



