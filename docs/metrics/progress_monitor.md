# Documentation for Progress_Monitor (progress_monitor.py)

## class ModelAssessment
### \_\_init\_\_
```py

def __init__(self, data, model, fantasy_steps=10, num_fantasy_particles=None, beta_std=0)

```



Create a ModelAssessment object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;data (tensor ~ (num_samples, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;model (BoltzmannMachine)<br />&nbsp;&nbsp;&nbsp;&nbsp;fantasy_steps (int; optional)<br />&nbsp;&nbsp;&nbsp;&nbsp;num_fantasy_particles (int; optional)<br />&nbsp;&nbsp;&nbsp;&nbsp;beta_std (float; optional)


### comparison
```py

def comparison(self, func, numpy=True)

```



Compare a function computed from the data and model states.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;func (callable): func: State -> tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;numpy (optional; bool): return the arrays in numpy form if true<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;data (tensor ~ 1D), model (tensor ~ 1D),<br />&nbsp;&nbsp;&nbsp;&nbsp;correlation (float), root mean squared error (float)


### sample\_data
```py

def sample_data(self, sample_indices, layer=0, func=None)

```



Select a subset samples from the data state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;sample_indices (tensor): list of indices<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (optional; int): the layer to get from the state<br />&nbsp;&nbsp;&nbsp;&nbsp;func (optional; callable): a function to apply to the units<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;data: List[tensor]


### sample\_model
```py

def sample_model(self, sample_indices, layer=0, func=None)

```



Select a subset samples from the model state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;sample_indices (tensor): list of indices<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (optional; int): the layer to get from the state<br />&nbsp;&nbsp;&nbsp;&nbsp;func (optional; callable): a function to apply to the units<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;model: List[tensor]


### sample\_reconstructions
```py

def sample_reconstructions(self, sample_indices, layer=0, func=None)

```



Select a subset samples from the model state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;sample_indices (tensor): list of indices<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (optional; int): the layer to get from the state<br />&nbsp;&nbsp;&nbsp;&nbsp;func (optional; callable): a function to apply to the units<br /><br />Returns:<br /><br />&nbsp;&nbsp;&nbsp;&nbsp;data: List[tensor], reconstructions: List[tensor]




## class ProgressMonitor
Monitor the progress of training by computing statistics on the<br />validation set.
### \_\_init\_\_
```py

def __init__(self, generator_metrics=[<paysage.metrics.generator_metrics.ReconstructionError object>, <paysage.metrics.generator_metrics.EnergyCoefficient object>, <paysage.metrics.generator_metrics.HeatCapacity object>, <paysage.metrics.generator_metrics.WeightSparsity object>, <paysage.metrics.generator_metrics.WeightSquare object>, <paysage.metrics.generator_metrics.KLDivergence object>, <paysage.metrics.generator_metrics.ReverseKLDivergence object>])

```



Create a progress monitor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;metrics (list[metric object]): list of metrics objects to compute with<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;ProgressMonitor


### batch\_update
```py

def batch_update(self, assessment)

```



Update the metrics on a batch.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### check\_save\_conditions
```py

def check_save_conditions(self, model)

```



Checks any save conditions.<br />Each check will save the model if it passes.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;model (paysage.models model): generative model<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### epoch\_update
```py

def epoch_update(self, batch, generator, fantasy_steps=10, store=False, show=False, filter_none=True, reset=True)

```



Outputs metric stats, and returns the metric dictionary<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch (paysage.batch object): data batcher<br />&nbsp;&nbsp;&nbsp;&nbsp;generator (paysage.models model): generative model<br />&nbsp;&nbsp;&nbsp;&nbsp;fantasy_steps (int): num steps to sample generator for fantasy particles<br />&nbsp;&nbsp;&nbsp;&nbsp;store (bool): if true, store the metrics in a list<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and check if the model should be saved<br />&nbsp;&nbsp;&nbsp;&nbsp;show (bool): if true, print the metrics to the screen<br />&nbsp;&nbsp;&nbsp;&nbsp;filter_none (bool): remove none values from metric output<br />&nbsp;&nbsp;&nbsp;&nbsp;reset (bool): reset the metrics on epoch update<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;metdict (dict): an ordered dictionary with the metrics


### get\_metric\_dict
```py

def get_metric_dict(self, filter_none=True)

```



Get the metrics in dictionary form.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;filter_none (bool): remove none values from metric output<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;OrderedDict


### plot\_metrics
```py

def plot_metrics(self, filename=None, show=True)

```



Plot the metric memory.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;filename (optional; str)<br />&nbsp;&nbsp;&nbsp;&nbsp;show (optional; bool)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### reset\_metrics
```py

def reset_metrics(self)

```



Reset the state of the metrics.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the metrics in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### save\_best
```py

def save_best(self, filename, metric, extremum='min')

```



Save the model when a given metric is extremal.<br />The filename will have the extremum and metric name appended,<br />&nbsp;&nbsp;&nbsp;&nbsp;e.g. "_min_EnergyCoefficient".<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies save_conditions in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;filename (str): the filename.<br />&nbsp;&nbsp;&nbsp;&nbsp;metric (str): the metric name.<br />&nbsp;&nbsp;&nbsp;&nbsp;extremum (str): "min" or "max"<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### save\_every
```py

def save_every(self, filename, epoch_period=1)

```



Save the model every N epochs.<br />The filename will have "_epoch<N>" appended.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies save_conditions in place.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;filename (str): the filename. "_epoch<N>" will be appended.<br />&nbsp;&nbsp;&nbsp;&nbsp;epoch_period (int): the period for saving the model. For example,<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if epoch_period=2, the model is saved on epochs 2, 4, 6, ...<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## class OrderedDict
Dictionary that remembers insertion order

