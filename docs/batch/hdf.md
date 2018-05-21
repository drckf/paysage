# Documentation for Hdf (hdf.py)

## class HDFtable
Serves up minibatches from a single table in an HDFStore.<br />The data should probably be randomly shuffled<br />if being used to train a model.
### \_\_init\_\_
```py

def __init__(self, filename, key, batch_size, transform=<paysage.preprocess.Transformation object>, combine_frames=False)

```



Creates an iterator that can pull minibatches from an HDFStore.<br />Works on a single table.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;filename (str): the HDFStore file to read from.<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the key of the table to read from.<br />&nbsp;&nbsp;&nbsp;&nbsp;batch_size (int): the minibatch size.<br />&nbsp;&nbsp;&nbsp;&nbsp;transform (Transformation): the transform function to apply to the data.<br />&nbsp;&nbsp;&nbsp;&nbsp;combine_frames (optional; bool): datasets with too many columns<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;have to be divided into chunks. These chunks are stored as<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;frames in the hdf5 file.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;An HDFtable instance.


### close
```py

def close(self) -> None

```



Close the HDFStore.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### get
```py

def get(self)

```



Get the next minibatch.<br />Will raise a StopIteration if the end of the data is reached.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the minibatch of data.


### get\_by\_index
```py

def get_by_index(self, index)

```



Get the next minibatch by index.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;index (Listable): the index values to select.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;tensor: the minibatch of data.


### reset\_generator
```py

def reset_generator(self) -> None

```



Reset the generator.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;mode (str): the mode, 'train', 'validate', or 'all'.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### set\_parameters\_with\_test
```py

def set_parameters_with_test(self)

```



Set the batch-dependent parameters with a test call to get.<br />This allows to account for preprocess functions that transform<br />the output batch size, rows, or columns.<br /><br />Notes:<br />&nbsp;&nbsp;&nbsp;&nbsp;Modifies output_batch_size attribute in place, resets the generator.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None




## functions

### maybe\_int
```py

def maybe_int(x)

```


