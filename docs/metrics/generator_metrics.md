# Documentation for Generator_Metrics (generator_metrics.py)

## class JensenShannonDivergence
Compute the JS divergence between two samples using the method of:<br /><br />"Divergence Estimation for Multidimensional Densities Via k-Nearest Neighbor<br />Distances"<br />by Qing Wang, Sanjeev R. Kulkarni and Sergio Verdú<br /><br />JS(P || Q) = 1/2*KL(P || 1/2(P + Q)) + 1/2*KL(Q || 1/2(P + Q))<br /><br />p ~ model samples<br />q ~ data samples<br /><br />We provide the option to divide out by the dimension of the dataset.
### \_\_init\_\_
```py

def __init__(self, k=5, name='JensenShannonDivergence', divide_dimension=True)

```



Create JensenShannonKLDivergence object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;k (int; optional): which nearest neighbor to use<br />&nbsp;&nbsp;&nbsp;&nbsp;name (str; optional): metric name<br />&nbsp;&nbsp;&nbsp;&nbsp;divide_dimension (bool; optional): whether to divide the divergence<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;by the number of dimensions<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;JensenShannonDivergence object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the JS divergence using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses data_state and model_state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the reverse JS divergence estimate.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;JS divergence estimate (float)




## class ReconstructionError
Compute the root-mean-squared error between observations and their<br />reconstructions using minibatches, rescaled by the minibatch variance.
### \_\_init\_\_
```py

def __init__(self, name='ReconstructionError')

```



Create a ReconstructionError object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;name (str; optional): metric name<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;ReconstructionError


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the reconstruction error using a batch<br />of observations and a batch of reconstructions.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses data_state and reconstructions<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the reconstruction error.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;reconstruction error (float)




## class ReverseKLDivergence
Compute the reverse KL divergence between two samples using the method of:<br /><br />"Divergence Estimation for Multidimensional Densities Via k-Nearest Neighbor<br />Distances"<br />by Qing Wang, Sanjeev R. Kulkarni and Sergio Verdú<br /><br />KL(P || Q) = \int dx p(x) log(p(x)/q(x))<br /><br />p ~ model samples<br />q ~ data samples<br /><br />We provide the option to divide out the dimension.
### \_\_init\_\_
```py

def __init__(self, k=5, name='ReverseKLDivergence', divide_dimension=True)

```



Create ReverseKLDivergence object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;k (int; optional): which nearest neighbor to use<br />&nbsp;&nbsp;&nbsp;&nbsp;name (str; optional): metric name<br />&nbsp;&nbsp;&nbsp;&nbsp;divide_dimension (bool; optional): whether to divide the divergence<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;by the number of dimensions<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;ReverseKLDivergence object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the reverse KL divergence using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses data_state and model_state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the reverse KL divergence estimate.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;reverse KL divergence estimate (float)




## class EnergyCoefficient
Compute a normalized energy distance between two distributions using<br />minibatches of sampled configurations.<br /><br />Szekely, G.J. (2002)<br />E-statistics: The Energy of Statistical Samples.<br />Technical Report BGSU No 02-16.
### \_\_init\_\_
```py

def __init__(self, name='EnergyCoefficient')

```



Create EnergyCoefficient object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;EnergyCoefficient object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the energy coefficient using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses data_state and model_state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the energy coefficient.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;energy coefficient (float)




## class TAPLogLikelihood
Compute the log likelihood of the data using the TAP2 approximation of -lnZ_model
### \_\_init\_\_
```py

def __init__(self, num_samples=2, name='TAPLogLikelihood')

```



Create TAPLogLikelihood object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_samples (int): number of samples to average over<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the TAP free energy and the marginal free energy<br /> (actually the average per sample)<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses model<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the average TAP log likelihood.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;the average TAP log likelihood (float)




## class WeightSparsity
Compute the weight sparsity of the model as the formula<br /><br />p = \sum_j(\sum_i w_ij^2)^2/\sum_i w_ij^4<br /><br />Tubiana, J., Monasson, R. (2017)<br />Emergence of Compositional Representations in Restricted Boltzmann Machines,<br />PRL 118, 138301 (2017)
### \_\_init\_\_
```py

def __init__(self, name='WeightSparsity')

```



Create WeightSparsity object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;WeightSparsity object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Compute the weight sparsity of the model<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;If the value already exists, it is not updated.<br />&nbsp;&nbsp;&nbsp;&nbsp;Call reset() between model updates.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses model<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the weight sparsity.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;weight sparsity (float)




## class TAPFreeEnergy
Compute the TAP2 free energy of the model seeded from some number of<br />random magnetizations.  This value approximates -lnZ_model
### \_\_init\_\_
```py

def __init__(self, num_samples=2, name='TAPFreeEnergy')

```



Create TAPFreeEnergy object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_samples (int): number of samples to average over<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the TAP free energy.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses model<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the average TAP free energy.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;the average TAP free energy (float)




## class FrechetScore
Compute the Frechet Score between two samples. Based on an idea from:<br /><br />"GANs Trained by a Two Time-Scale Update Rule Converge to a<br />Local Nash Equilibrium"<br />by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler<br />Sepp Hochreiter<br /><br />but without the inception network.
### \_\_init\_\_
```py

def __init__(self, name='FrechetScore')

```



Create FrechetScore object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;FrechetScore object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the Frechet Score using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses data_state and model_state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the Frechet Score estimate.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Frechet Score estimate (float)




## class HeatCapacity
Compute the heat capacity of the model per parameter.<br /><br />We take the HC to be the second cumulant of the energy, or alternately<br />the negative second derivative with respect to inverse temperature of<br />the Gibbs free energy.  In order to estimate this quantity we perform<br />Gibbs sampling starting from random samples drawn from the visible layer's<br />distribution.  This is rescaled by the number of units parameters in the model.
### \_\_init\_\_
```py

def __init__(self, name='HeatCapacity')

```



Create HeatCapacity object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the heat capacity.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses model and model_state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the heat capacity.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;heat capacity (float)




## class KLDivergence
Compute the KL divergence between two samples using the method of:<br /><br />"Divergence Estimation for Multidimensional Densities Via k-Nearest Neighbor<br />Distances"<br />by Qing Wang, Sanjeev R. Kulkarni and Sergio Verdú<br /><br />KL(P || Q) = \int dx p(x) log(p(x)/q(x))<br /><br />p ~ data samples<br />q ~ model samples<br /><br />We provide the option to remove dependence on dimension, true by default.
### \_\_init\_\_
```py

def __init__(self, k=5, name='KLDivergence', divide_dimension=True)

```



Create KLDivergence object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;k (int; optional): which nearest neighbor to use<br />&nbsp;&nbsp;&nbsp;&nbsp;name (str; optional): metric name<br />&nbsp;&nbsp;&nbsp;&nbsp;divide_dimension (bool; optional): whether to divide the divergence<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;by the number of dimensions<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;KLDivergence object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Update the estimate for the KL divergence using a batch<br />of observations and a batch of fantasy particles.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses data_state and model_state<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the KL divergence estimation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;KL divergence estimation (float)




## class WeightSquare
Compute the mean squared weights of the model per hidden unit<br /><br />w2 = 1/(#hidden units)*\sum_ij w_ij^2<br /><br />Tubiana, J., Monasson, R. (2017)<br />Emergence of Compositional Representations in Restricted Boltzmann Machines,<br />PRL 118, 138301 (2017)
### \_\_init\_\_
```py

def __init__(self, name='WeightSquare')

```



Create WeightSquare object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;WeightSquare object


### reset
```py

def reset(self) -> None

```



Reset the metric to its initial state.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### update
```py

def update(self, assessment) -> None

```



Compute the weight square of the model.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;If the value already exists, it is not updated.<br />&nbsp;&nbsp;&nbsp;&nbsp;Call reset() between model updates.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;assessment (ModelAssessment): uses model<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### value
```py

def value(self) -> float

```



Get the value of the weight sparsity.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;weight sparsity (float)



