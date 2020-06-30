# Documentation for Shuffle (shuffle.py)

## class TableStatistics
Stores basic statistics about a table.
### \_\_init\_\_
```py

def __init__(self, store, key)

```



Constructor.  Records the basic statistics.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;store (HDFStore): the store with the table.<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the key of the table.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;A TableStatistics instance.


### chunksize
```py

def chunksize(self, allowed_mem)

```



Returns the sample count that will fit in allowed_mem,<br />given the shape of the table.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;allowed_mem (float): allowed memory in GiB<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;chunksize (int): the chunk size that will fit in the allowed memory.




## class DataShuffler
Shuffles data in an HDF5 file.  Memory is managed.<br />Synchronized shuffling between tables (with matching numbers of rows).<br /><br />Each table is shuffled with the following algorithm:<br />&nbsp;&nbsp;&nbsp;&nbsp;- The table is sequentially divided into chunks.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Each is shuffled and saved to a temporary file.<br />&nbsp;&nbsp;&nbsp;&nbsp;- Pieces of each chunk are read in and shuffled together,<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Then written to the target file.<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This is done in a memory-managed way that preserves randomness.
### \_\_init\_\_
```py

def __init__(self, filename, shuffled_filename, allowed_mem=1, complevel=5, seed=137)

```



Constructor.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;filename (str): the filename of the data to be shuffled.<br />&nbsp;&nbsp;&nbsp;&nbsp;shuffled_filename (str): the filename to write the shuffled data to.<br />&nbsp;&nbsp;&nbsp;&nbsp;allowed_mem (float): the allowed memory footprint in GiB.<br />&nbsp;&nbsp;&nbsp;&nbsp;complevel (int): the compression level used by pandas.<br />&nbsp;&nbsp;&nbsp;&nbsp;seed (int): the random number seed for the shuffler.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;A DataShuffler instance.


### divide\_table\_into\_chunks
```py

def divide_table_into_chunks(self, key)

```



Divides a table into chunks, each with their own table.<br />Shuffles the chunked tables.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the key of the table to chunk.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;num_chunks (int): the number of chunks the table was divided into.<br />&nbsp;&nbsp;&nbsp;&nbsp;chunk_keys (List[str]): the keys of the chunks in the temporary file.<br />&nbsp;&nbsp;&nbsp;&nbsp;chunk_counts (List[int]): the number of samples in each of the chunks.


### reassemble\_table
```py

def reassemble_table(self, key, num_chunks, chunk_keys, chunk_counts)

```



Takes a set of chunked tables and rebuilds the shuffled table.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the key of the table to shuffle.<br />&nbsp;&nbsp;&nbsp;&nbsp;num_chunks (int): the number of chunks the table was divided into.<br />&nbsp;&nbsp;&nbsp;&nbsp;chunk_keys (List[str]): the keys of the chunks in the temporary file.<br />&nbsp;&nbsp;&nbsp;&nbsp;chunk_counts (List[int]): the number of samples in each of the chunks.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### shuffle
```py

def shuffle(self)

```



Shuffles all the tables in the HDFStore.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;None<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None


### shuffle\_table
```py

def shuffle_table(self, key)

```



Shuffle a table in the HDFStore, write to a new file.<br /><br />Args:<br />&nbsp;&nbsp;&nbsp;&nbsp;key (str): the key of the table to shuffle.<br /><br />Returns:<br />&nbsp;&nbsp;&nbsp;&nbsp;None



