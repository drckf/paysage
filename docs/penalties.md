# Documentation for Penalties (penalties.py)

## class l1_adaptive_decay_penalty_2
Modified form of the l1 penalty which regularizes more for weight rows<br />with larger coupling to target layer by way of a quadratic power<br /><br />Tubiana, J., Monasson, R. (2017)<br />Emergence of Compositional Representations in Restricted Boltzmann Machines,<br />PRL 118, 138301 (2017), Supplemental Material I.D<br /><br />Note: expects to operate on a tensor with two degrees of freedom (eg. a weight matrix)
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create an adaptive l1 penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;l1_adaptive_decay_penalty_2


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class trivial_penalty
A penalty that does nothing.
### \_\_init\_\_
```py

def __init__(self, penalty_unused=0, slice_tuple_unused=(slice(None, None, None),))

```



Create a base trivial penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty_unused (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple_unused (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;trivial_penalty


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class exp_l2_penalty
Puts an l2 penalty on the exponentiated parameters.<br />Useful when the parameters are represented in a logarithmic space.<br />For example, encouraging the variances of a GaussianLayer to be small.
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create an exp l2 penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;exp_l2_penalty


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class logdet_penalty
Penalty acts on the logarithm of the determinant of a matrix.<br />Discourages the matrix from becoming singular.
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create a logdet_penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;logdet_penalty


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class log_penalty
A logarithmic penalty forces the parameters to be positive.
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create a log penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;log_penalty


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class l1_penalty
An l1 penalty encourages small values of the parameters,<br />and tends to produce solutions that are more sparse than an l2 penalty.<br />Also known as "lasso".
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create an l1 penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;l1_penalty


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class l2_penalty
An L2 penalty encourages small values of the parameters.<br />Also known as a "ridge" penalty.
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create an l2 penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;l2_penalty


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class log_norm
An RBM with n visible units and m hidden units has an (n, m) weight matrix.<br />The log norm penalty discourages any of the m columns of the weight matrix<br />from having zero norm.
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create an log_norm.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;log_norm


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class Penalty
Base penalty class.<br />Derived classes should define `value` and `grad` functions.
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),))

```



Create a base Penalty object.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Penalty


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## class l2_norm
An RBM with n visible units and m hidden units has an (n, m) weight matrix.<br />The l2 norm penalty encourages the columns of the weight matrix to have<br />norms that are close to the target value.<br /><br />"On Training Deep Boltzmann Machines"<br />by Guillaume Desjardins, Aaron Courville, Yoshua Bengio<br />http://arxiv.org/pdf/1203.4416.pdf
### \_\_init\_\_
```py

def __init__(self, penalty, slice_tuple=(slice(None, None, None),), target=1)

```



Create an l2_norm penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;penalty (float): strength of the penalty<br />&nbsp;&nbsp;&nbsp;&nbsp;slice_tuple (slice): list of slices that define the parts of the<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tensor to which the penalty will be applied<br />&nbsp;&nbsp;&nbsp;&nbsp;target (optional; float): the shrinkage target<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;l2_norm


### get\_config
```py

def get_config(self)

```



Returns a config for the penalty.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;dict


### grad
```py

def grad(self, tensor)

```



The value of the gradient of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### value
```py

def value(self, tensor)

```



The value of the penalty function on a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;float




## functions

### from\_config
```py

def from_config(config)

```



Builds an instance from a config.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;config (List or dict)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Penalty

