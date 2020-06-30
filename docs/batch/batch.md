# Documentation for Batch (batch.py)

## class Batch
Serves up minibatches from train and validation data sources.<br />The data should probably be randomly shuffled<br />if being used to train a model.
### \_\_init\_\_
```py

def __init__(self, batch_dictionary)

```



Holds data sources with iterators that can pull minibatches.<br />The train and validate batchers must have methods:<br />&nbsp;&nbsp;&nbsp;&nbsp;- set_parameters_with_test<br />&nbsp;&nbsp;&nbsp;&nbsp;- get<br />&nbsp;&nbsp;&nbsp;&nbsp;- get_by_index<br />&nbsp;&nbsp;&nbsp;&nbsp;- reset_generator<br />&nbsp;&nbsp;&nbsp;&nbsp;- close<br /><br />This object will hold the same dataset-level attributes<br />as the training batcher.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;batch_dictionary (Dict[str: InMemoryTable/HDFtable])<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;A Batch instance.


### close
```py

def close(self, mode: str = 'all') -> None

```



Close the data sources.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mode (str): the mode, 'train', 'validate', or 'all'.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### get
```py

def get(self, mode: str)

```



Get the next minibatch.<br />Will raise a StopIteration if the end of the data is reached.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mode (str): the mode to read, 'train' or 'validate'.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the minibatch of data.


### get\_by\_index
```py

def get_by_index(self, mode, index)

```



Get the next minibatch by index.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mode (str): the mode to read, 'train' or 'validate'.<br />&nbsp;&nbsp;&nbsp;&nbsp;index (Listable): the index values to select.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the minibatch of data.


### get\_transforms
```py

def get_transforms(self)

```



Return the transform functions.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None:<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;Dict[Callable]


### reset\_generator
```py

def reset_generator(self, mode: str) -> None

```



Reset the generator.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mode (str): the mode, 'train', 'validate', or 'all'.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_parameters\_with\_test
```py

def set_parameters_with_test(self, mode: str = 'all')

```



Set the batch-dependent parameters with a test call to get.<br />This allows to account for preprocess functions that transform<br />the output batch size, number of steps, rows, or columns.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies batch attributes inplace.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mode (str): the mode, 'train', 'validate', or 'all'.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_transforms
```py

def set_transforms(self, transforms)

```



Set the transform functions.<br /><br />Note:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifes the batch[key].transforms attributes for key \in [train, validate]!<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;transforms (Dict[Callable])<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## functions

### in\_memory\_batch
```py

def in_memory_batch(tensor, batch_size, train_fraction=0.9, transform=<paysage.preprocess.Transformation object>)

```



Utility function to create a Batch object from a tensor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor (tensors): the tensor to batch.<br />&nbsp;&nbsp;&nbsp;&nbsp;batch_size (int): the (common) batch size.<br />&nbsp;&nbsp;&nbsp;&nbsp;train_fraction (float): the fraction of data to use as training data.<br />&nbsp;&nbsp;&nbsp;&nbsp;transform (callable): the (common) transform function.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;data (Batch): the batcher.


### split\_tensor
```py

def split_tensor(tensor, split_fraction)

```



Split a list of tensors into two parts into two fractions.<br />Assumes the tensors are all the same length.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors (List[tensors]): the tensors to split.<br />&nbsp;&nbsp;&nbsp;&nbsp;split_fraction (double): the fraction of the dataset to split at.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors_part1 (tensor): the first part of the tensor.<br />&nbsp;&nbsp;&nbsp;&nbsp;tensors_part2 (tensor): the second part of the tensor.

