# Documentation for Preprocess (preprocess.py)

## class Transformation
### \_\_init\_\_
```py

def __init__(self, function=<function do_nothing>, args=None, kwargs=None)

```



Create a transformation that operates on a list of tensors.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;function (optional; callable)<br />&nbsp;&nbsp;&nbsp;&nbsp;args (optional; List)<br />&nbsp;&nbsp;&nbsp;&nbsp;kwargs (optional; Dict)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Transformation


### compute
```py

def compute(self, tensor)

```



Apply the transformation to a single tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor


### get\_config
```py

def get_config(self)

```



Get the configuration of a transformation.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Dict




## functions

### binarize\_color
```py

def binarize_color(tensor)

```



Scales an int8 "color" value to [0, 1].<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))


### l1\_normalize
```py

def l1_normalize(tensor)

```



Divide the rows of a tensor by their L1 norms.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))


### l2\_normalize
```py

def l2_normalize(tensor)

```



Divide the rows of a tensor by their L2 norms.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))


### one\_hot
```py

def one_hot(data, category_list)

```



Convert a categorical variable into a one-hot code.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;data (tensor (num_samples, 1)): a column of the data matrix that is categorical<br />&nbsp;&nbsp;&nbsp;&nbsp;category_list: the list of categories<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;one-hot encoded data (tensor (num_samples, num_categories))


### scale
```py

def scale(tensor, denominator=1)

```



Rescale the values in a tensor by the denominator.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))<br />&nbsp;&nbsp;&nbsp;&nbsp;denominator (optional; float)<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor (num_samples, num_units))

