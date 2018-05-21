# Documentation for In_Memory (in_memory.py)

## class InMemoryTable
Serves up minibatches from a tensor held in memory.<br />The data should probably be randomly shuffled<br />if being used to train a model.
### \_\_init\_\_
```py

def __init__(self, tensor, batch_size, transform=<paysage.preprocess.Transformation object>)

```



Creates iterators that can pull minibatches<br />from a list of in-memory arrays.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensors): the array to batch<br />&nbsp;&nbsp;&nbsp;&nbsp;batch_size (int): the minibatch size<br />&nbsp;&nbsp;&nbsp;&nbsp;transform (Transformation): the transform function to apply to the data<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;An InMemoryTable instance.


### close
```py

def close(self) -> None

```



Frees the tensor.


### get
```py

def get(self)

```



Get the next minibatch.<br />Will raise a StopIteration if the end of the data is reached.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the minibatch of data.


### get\_by\_index
```py

def get_by_index(self, index)

```



Get the next minibatch by index.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;index (tensor): the index values to select.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the minibatch of data.


### reset\_generator
```py

def reset_generator(self) -> None

```



Reset the generator.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_parameters\_with\_test
```py

def set_parameters_with_test(self)

```



Set the batch-dependent parameters with a test call to get.<br />This allows to account for preprocess functions that transform<br />the output batch size, number of steps, rows, or columns.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies output_batch_size.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## functions

### inclusive\_slice
```py

def inclusive_slice(tensor, start, stop, step)

```



Generator yielding progressive inclusive slices from a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensors): the tensors to minibatch.<br />&nbsp;&nbsp;&nbsp;&nbsp;start (int): the start index.<br />&nbsp;&nbsp;&nbsp;&nbsp;stop (int): the stop index.<br />&nbsp;&nbsp;&nbsp;&nbsp;step (int): the minibatch size.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensor): a minibatch of tensors.

