# Documentation for Pca (pca.py)

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




## class PCA
### \_\_init\_\_
```py

def __init__(self, num_components, stepsize=0.001)

```



Computes the principal components of a dataset using stochastic gradient<br />descent.<br /><br />Arora, Raman, et al.<br />"Stochastic optimization for PCA and PLS."<br />Communication, Control, and Computing (Allerton), 2012<br />50th Annual Allerton Conference on. IEEE, 2012.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_components (int): The number of directions to extract.<br />&nbsp;&nbsp;&nbsp;&nbsp;stepsize (optional): Learning rate schedule.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;PCA


### compute\_error\_on\_batch
```py

def compute_error_on_batch(self, tensor)

```



Compute the reconstruction error || X - X W W^T ||^2.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float


### compute\_validation\_error
```py

def compute_validation_error(self, batch)

```



Compute the root-mean-squared reconstruction error from the<br />validation set.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch: a batch object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float


### compute\_validation\_kld
```py

def compute_validation_kld(self, batch)

```



Compute the KL divergence between the pca distribution and the<br />distribution of the validation set.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch: a batch object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float


### compute\_validation\_rkld
```py

def compute_validation_rkld(self, batch)

```



Compute the Reverse KL divergence between the pca distribution and the<br />distribution of the validation set.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch: a batch object<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float


### project
```py

def project(self, tensor)

```



Project a tensor onto the principal components.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_components)


### sample\_pca
```py

def sample_pca(self, n)

```



Sample from the multivariate Gaussian represented by the pca<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;n (int): number of samples<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;samples (tensor (n, num_units))


### save
```py

def save(self, store: pandas.io.pytables.HDFStore, num_components_save: int=None) -> None

```



Save the PCA transform in an HDFStore.<br />Allows to save only the first num_components_save.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Performs an IO operation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (pandas.HDFStore)<br />&nbsp;&nbsp;&nbsp;&nbsp;num_components_save (int): the number of principal components to save.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If None, all are saved.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### train\_on\_batch
```py

def train_on_batch(self, tensor, grad_steps=1, orthogonalize=False)

```



Update the principal components using stochastic gradient descent.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the PCA.W attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units): a batch of data<br />&nbsp;&nbsp;&nbsp;&nbsp;grad_steps (int): the number of gradient steps to make<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### transform
```py

def transform(self, tensor)

```



Transform a tensor by removing the global mean and projecting.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_components)


### update\_variance\_on\_batch
```py

def update_variance_on_batch(self, tensor)

```



Update the variances along the principle directions.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the PCA.var_cal attribute in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (num_samples, num_units)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None



