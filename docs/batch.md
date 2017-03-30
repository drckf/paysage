# Documentation for Batch (batch.py)

## class TableStatistics
TableStatistics<br />Stores basic statistics about a table.
### \_\_init\_\_
```py

def __init__(self, store, key)

```



Initialize self.  See help(type(self)) for accurate signature.


### chunksize
```py

def chunksize(self, allowed_mem)

```



chunksize<br />Returns the sample count that will fit in allowed_mem,<br />given the shape of the table.




## class DataShuffler
DataShuffler<br />Shuffles data in an HDF5 file.<br />Synchronized shuffling between tables (with matching numbers of rows).
### \_\_init\_\_
```py

def __init__(self, filename, shuffled_filename, allowed_mem=1, complevel=5, seed=137)

```



Initialize self.  See help(type(self)) for accurate signature.


### divide\_table\_into\_chunks
```py

def divide_table_into_chunks(self, key)

```



divide_table_into_chunks<br />Divides a table into chunks, each with their own table.<br />Shuffles the chunked tables.


### reassemble\_table
```py

def reassemble_table(self, key, num_chunks, chunk_keys, chunk_counts)

```



reassemble_table<br />Takes a set of chunked tables and rebuilds the shuffled table.


### shuffle
```py

def shuffle(self)

```



shuffle<br />Shuffles all the tables in the HDFStore.


### shuffle\_table
```py

def shuffle_table(self, key)

```



shuffle_table<br />Shuffle a table in the HDFStore, write to a new file.




## class Batch
Batch<br />Serves up minibatches from an HDFStore.<br />The validation set is taken as the last (1 - train_fraction)<br />samples in the store.<br />The data should probably be randomly shuffled if being used to<br />train a non-recurrent model.
### \_\_init\_\_
```py

def __init__(self, filename, key, batch_size, train_fraction=0.9, transform=<function float_tensor at 0x1264a3158>)

```



Initialize self.  See help(type(self)) for accurate signature.


### close
```py

def close(self)

```



### get
```py

def get(self, mode)

```



### num\_validation\_samples
```py

def num_validation_samples(self)

```



### reset\_generator
```py

def reset_generator(self, mode)

```





## functions

### binarize\_color
```py

def binarize_color(tensor)

```



binarize_color<br />Scales an int8 "color" value to [0, 1].  Converts to float32.


### binary\_to\_ising
```py

def binary_to_ising(tensor)

```



binary_to_ising<br />Scales a [0, 1] value to [-1, 1].  Converts to float32.


### color\_to\_ising
```py

def color_to_ising(tensor)

```



color_to_ising<br />Scales an int8 "color" value to [-1, 1].  Converts to float32.


### do\_nothing
```py

def do_nothing(tensor)

```



### scale
```py

def scale(tensor, denominator)

```


