# Documentation for Preprocess (preprocess.py)

## class partial
partial(func, *args, **keywords) - new function with partial application<br />of the given arguments and keywords.


## functions

### binarize\_color
```py

def binarize_color(tensor)

```



Scales an int8 "color" value to [0, 1].<br /><br />Args:<br /> ~ tensor<br /><br />Returns:<br /> ~ float tensor


### binary\_to\_ising
```py

def binary_to_ising(tensor)

```



Scales a [0, 1] value to [-1, 1].<br /><br />Args:<br /> ~ tensor<br /><br />Returns:<br /> ~ float tensor


### color\_to\_ising
```py

def color_to_ising(tensor)

```



Scales an int8 "color" value to [-1, 1].<br /><br />Args:<br /> ~ tensor<br /><br />Returns:<br /> ~ float tensor


### do\_nothing
```py

def do_nothing(tensor)

```



Identity function.<br /><br />Args:<br /> ~ Anything.<br /><br />Returns:<br /> ~ Anything.


### l1\_normalize
```py

def l1_normalize(tensor)

```



Divide the rows of the tensor by their L1 norms.<br /><br />Args:<br /> ~ tensor (num_samples, num_units)<br /><br />Returns:<br /> ~ tensor (num_samples, num_units)


### l2\_normalize
```py

def l2_normalize(tensor)

```



Divide the rows of the tensory by their L2 norms.<br /><br />Args:<br /> ~ tensor (num_samples, num_units)<br /><br />Returns:<br /> ~ tensor (num_samples, num_units)


### one\_hot
```py

def one_hot(data, category_list)

```



Convert a categorical variable into a one-hot code.<br /><br />Args:<br /> ~ data (tensor (num_samples, 1)): a column of the data matrix that is categorical<br /> ~ category_list: the list of categories<br /><br />Returns:<br /> ~ one-hot encoded data (tensor (num_samples, num_categories))


### scale
```py

def scale(tensor, denominator)

```



Rescale the values in a tensor by the denominator.<br /><br />Args:<br /> ~ tensor: A tensor.<br /> ~ denominator (float)<br /><br />Returns:<br /> ~ float tensor

