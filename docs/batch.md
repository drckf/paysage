# Documentation for Batch (batch.py)

## class TableStatistics
Stores basic statistics about a table.
### \_\_init\_\_
```py

def __init__(self, store, key)

```



Initialize self.  See help(type(self)) for accurate signature.


### chunksize
```py

def chunksize(self, allowed_mem)

```



Returns the sample count that will fit in allowed_mem,<br />given the shape of the table.




## class InMemoryBatch
Serves up minibatches from a tensor held in memory.<br />The validation set is taken as the last (1 - train_fraction)<br />samples in the store.<br />The data should probably be randomly shuffled if being used to<br />train a non-recurrent model.
### \_\_init\_\_
```py

def __init__(self, tensor, batch_size, train_fraction=0.9, transform=<function do_nothing at 0x11e81d510>)

```



Initialize self.  See help(type(self)) for accurate signature.


### close
```py

def close(self) -> None

```



### get
```py

def get(self, mode: str)

```



### get\_by\_index
```py

def get_by_index(self, index)

```



### num\_training\_batches
```py

def num_training_batches(self)

```



### num\_validation\_samples
```py

def num_validation_samples(self) -> int

```



### reset\_generator
```py

def reset_generator(self, mode: str) -> None

```





## class DataShuffler
Shuffles data in an HDF5 file.<br />Synchronized shuffling between tables (with matching numbers of rows).
### \_\_init\_\_
```py

def __init__(self, filename, shuffled_filename, allowed_mem=1, complevel=5, seed=137)

```



Initialize self.  See help(type(self)) for accurate signature.


### divide\_table\_into\_chunks
```py

def divide_table_into_chunks(self, key)

```



Divides a table into chunks, each with their own table.<br />Shuffles the chunked tables.


### reassemble\_table
```py

def reassemble_table(self, key, num_chunks, chunk_keys, chunk_counts)

```



Takes a set of chunked tables and rebuilds the shuffled table.


### shuffle
```py

def shuffle(self)

```



Shuffles all the tables in the HDFStore.


### shuffle\_table
```py

def shuffle_table(self, key)

```



Shuffle a table in the HDFStore, write to a new file.




## class HDFBatch
Serves up minibatches from an HDFStore.<br />The validation set is taken as the last (1 - train_fraction)<br />samples in the store.<br />The data should probably be randomly shuffled if being used to<br />train a non-recurrent model.
### \_\_init\_\_
```py

def __init__(self, filename, key, batch_size, train_fraction=0.9, transform=<function do_nothing at 0x11e81d510>)

```



Initialize self.  See help(type(self)) for accurate signature.


### close
```py

def close(self) -> None

```



### get
```py

def get(self, mode: str)

```



### get\_by\_index
```py

def get_by_index(self, index)

```



### num\_training\_batches
```py

def num_training_batches(self)

```



### num\_validation\_samples
```py

def num_validation_samples(self) -> int

```



### reset\_generator
```py

def reset_generator(self, mode: str) -> None

```





## functions

### inclusive\_slice
```py

def inclusive_slice(tensor, start, stop, step)

```


