#  Documentation for Constraints (constraints.py)

## functions

### diagonal
```py

def diagonal(tensor)

```



Set any off-diagonal entries of the input tensor to zero.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the input tensor in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### fixed\_column\_norm
```py

def fixed_column_norm(tensor)

```



Renormalize the tensor so that all of its columns have the same norm.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the input tensor in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### non\_negative
```py

def non_negative(tensor)

```



Set any negative entries of the input tensor to zero.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the input tensor in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### non\_positive
```py

def non_positive(tensor)

```



Set any positive entries of the input tensor to zero.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the input tensor in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### zero\_column
```py

def zero_column(tensor, index)

```



Set any entries of in the given column of the input tensor to zero.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the input tensor in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;index (int): index of the column to set to zero<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### zero\_mask
```py

def zero_mask(tensor, mask)

```



Set the given entries of the input tensor to zero.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the input tensor in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;mask: a binary mask of the same shape as tensor. entries where the mask<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;is 1 will be set to zero<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### zero\_row
```py

def zero_row(tensor, index)

```



Set any entries of in the given row of the input tensor to zero.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies the input tensor in place!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor<br />&nbsp;&nbsp;&nbsp;&nbsp;index (int): index of the row to set to zero<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None

