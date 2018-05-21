# Documentation for Methods (methods.py)

## class State
A State is a list of tensors that contains the states of the units<br />described by a model.<br /><br />For a model with L hidden layers, the tensors have shapes<br /><br />shapes = [<br />(num_samples, num_visible),<br />(num_samples, num_hidden_1),<br />            .<br />            .<br />            .<br />(num_samples, num_hidden_L)<br />]
### \_\_init\_\_
```py

def __init__(self, tensors)

```



Create a State object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors: a list of tensors<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;state object


### batch\_size
```py

def batch_size(self)

```



Get the batch size of the state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch size: int


### get\_visible
```py

def get_visible(self)

```



Extract the visible units<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;vis (tensor (num_samples, num_visible)): visible unit values.


### number\_of\_layers
```py

def number_of_layers(self)

```



Get the number of layers in the state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;number_of_layers (int)


### number\_of\_units
```py

def number_of_units(self, layer)

```



Get the number of units in a layer of the state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;layer (int)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;number_of_units (int)




## class TAP
### \_\_init\_\_
```py

def __init__(self, use_GD=True, init_lr=0.1, tolerance=0.01, max_iters=25, ratchet=False, decrease_on_neg=0.9, mean_weight=0.9, mean_square_weight=0.999)

```



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


### tap\_update
```py

def tap_update(self, vdata, model, positive_phase, negative_phase=None)

```



Compute the gradient using the Thouless-Anderson-Palmer (TAP)<br />mean field approximation.<br /><br />Modifications on the methods in<br /><br />Eric W Tramel, Marylou Gabrie, Andre Manoel, Francesco Caltagirone,<br />and Florent Krzakala<br />"A Deterministic and Generalized Framework for Unsupervised Learning<br />with Restricted Boltzmann Machines"<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vdata (tensor): observed visible units<br />&nbsp;&nbsp;&nbsp;&nbsp;model (BoltzmannMachine): model to train<br />&nbsp;&nbsp;&nbsp;&nbsp;positive_phase (Sampler): postive phase data sampler<br />&nbsp;&nbsp;&nbsp;&nbsp;negative_phase (Sampler): unused<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient object




## functions

### cd
```py

def cd(vdata, model, positive_phase, negative_phase)

```



Compute an approximation to the likelihood gradient using the CD-k<br />algorithm for approximate maximum likelihood inference.<br /><br />Hinton, Geoffrey E.<br />"Training products of experts by minimizing contrastive divergence."<br />Neural computation 14.8 (2002): 1771-1800.<br /><br />Carreira-Perpinan, Miguel A., and Geoffrey Hinton.<br />"On Contrastive Divergence Learning."<br />AISTATS. Vol. 10. 2005.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state of the sampler.<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the sampling attributes of the model<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vdata (tensor): observed visible units<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a model object<br />&nbsp;&nbsp;&nbsp;&nbsp;positive_phase: a sampler object<br />&nbsp;&nbsp;&nbsp;&nbsp;negative_phase: a sampler object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient


### contrastive\_divergence
```py

def contrastive_divergence(vdata, model, positive_phase, negative_phase)

```



Compute an approximation to the likelihood gradient using the CD-k<br />algorithm for approximate maximum likelihood inference.<br /><br />Hinton, Geoffrey E.<br />"Training products of experts by minimizing contrastive divergence."<br />Neural computation 14.8 (2002): 1771-1800.<br /><br />Carreira-Perpinan, Miguel A., and Geoffrey Hinton.<br />"On Contrastive Divergence Learning."<br />AISTATS. Vol. 10. 2005.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state of the sampler.<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the sampling attributes of the model<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vdata (tensor): observed visible units<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a model object<br />&nbsp;&nbsp;&nbsp;&nbsp;positive_phase: a sampler object<br />&nbsp;&nbsp;&nbsp;&nbsp;negative_phase: a sampler object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient


### pcd
```py

def pcd(vdata, model, positive_phase, negative_phase)

```



PCD-k algorithm for approximate maximum likelihood inference.<br /><br />Tieleman, Tijmen.<br />"Training restricted Boltzmann machines using approximations to the<br />likelihood gradient."<br />Proceedings of the 25th international conference on Machine learning.<br />ACM, 2008.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state of the sampler.<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the sampling attributes of the model<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vdata (List[tensor]): observed visible units<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a model object<br />&nbsp;&nbsp;&nbsp;&nbsp;positive_phase: a sampler object<br />&nbsp;&nbsp;&nbsp;&nbsp;negative_phase: a sampler object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient


### persistent\_contrastive\_divergence
```py

def persistent_contrastive_divergence(vdata, model, positive_phase, negative_phase)

```



PCD-k algorithm for approximate maximum likelihood inference.<br /><br />Tieleman, Tijmen.<br />"Training restricted Boltzmann machines using approximations to the<br />likelihood gradient."<br />Proceedings of the 25th international conference on Machine learning.<br />ACM, 2008.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the state of the sampler.<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the sampling attributes of the model<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;vdata (List[tensor]): observed visible units<br />&nbsp;&nbsp;&nbsp;&nbsp;model: a model object<br />&nbsp;&nbsp;&nbsp;&nbsp;positive_phase: a sampler object<br />&nbsp;&nbsp;&nbsp;&nbsp;negative_phase: a sampler object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;gradient

