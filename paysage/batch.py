import os
import numpy
import pandas
from . import backends as be

# ----- FUNCTIONS ----- #

def do_nothing(tensor):
    return tensor

def scale(tensor, denominator):
    return be.float_tensor(tensor) / denominator

def binarize_color(tensor):
    """binarize_color
       Scales an int8 "color" value to [0, 1].  Converts to float32.

    """
    return be.float_tensor(be.round(tensor/255))

def binary_to_ising(tensor):
    """binary_to_ising
       Scales a [0, 1] value to [-1, 1].  Converts to float32.

    """
    return 2.0 * be.float_tensor(tensor) - 1.0

def color_to_ising(tensor):
    """color_to_ising
       Scales an int8 "color" value to [-1, 1].  Converts to float32.

    """
    return binary_to_ising(binarize_color(tensor))

# ----- CLASSES ----- #

class Batch(object):
    """Batch
       Serves up minibatches from an HDFStore.
       The validation set is taken as the last (1 - train_fraction)
       samples in the store.
       The data should probably be randomly shuffled if being used to
       train a non-recurrent model.

    """
    def __init__(self, filename, key, batch_size,
                 train_fraction=0.9,
                 transform=be.float_tensor):

        assert callable(transform)
        self.transform = transform

        # open the store, get the dimensions of the keyed table
        self.store = pandas.HDFStore(filename, mode='r')
        self.key = key
        self.batch_size = batch_size
        self.ncols = self.store.get_storer(key).ncols
        self.nrows = self.store.get_storer(key).nrows
        self.split = int(numpy.ceil(train_fraction * self.nrows))

        # create iterators over the data for the train/validate sets
        self.iterators = {}
        self.iterators['train'] = self.store.select(key, stop=self.split,
                                                    iterator=True,
                                                    chunksize=self.batch_size)
        self.iterators['validate'] = self.store.select(key, start=self.split,
                                                       iterator=True,
                                                       chunksize=self.batch_size)

        self.generators = {mode: self.iterators[mode].__iter__()
                           for mode in self.iterators}

    def num_validation_samples(self):
        return self.nrows - self.split

    def close(self):
        self.store.close()

    def reset_generator(self, mode):
        if mode == 'train':
            self.generators['train'] = self.iterators['train'].__iter__()
        elif mode == 'validate':
            self.generators['validate'] = self.iterators['validate'].__iter__()
        else:
            self.generators = {mode: self.iterators[mode].__iter__()
                               for mode in self.iterators}

    def get(self, mode):
        try:
            vals = next(self.generators[mode]).as_matrix()
        except StopIteration:
            self.reset_generator(mode)
            raise StopIteration
        return self.transform(vals)



class TableStatistics(object):
    """TableStatistics
       Stores basic statistics about a table.

    """
    def __init__(self, store, key):
        self.key_store = store.get_storer(key)

        self.shape = (self.key_store.nrows, self.key_store.ncols)
        self.dtype = self.key_store.dtype[1].base
        self.itemsize = self.dtype.itemsize
        self.mem_footprint = numpy.prod(self.shape) * self.itemsize / 1024**3 # in GiB

    def chunksize(self, allowed_mem):
        """chunksize
           Returns the sample count that will fit in allowed_mem,
           given the shape of the table.

        """
        return int(self.shape[0] * allowed_mem / self.mem_footprint)


class DataShuffler(object):
    """DataShuffler
       Shuffles data in an HDF5 file.
       Synchronized shuffling between tables (with matching numbers of rows).

    """
    def __init__(self, filename, shuffled_filename,
                 allowed_mem=1,
                 complevel=5,
                 seed=137):
        self.filename = filename
        self.allowed_mem = allowed_mem # in GiB
        self.seed = seed # should keep this fixed for long-term determinism
        self.complevel = complevel
        self.complib = 'zlib'

        # get the keys and statistics
        self.store = pandas.HDFStore(filename, mode='r')
        self.keys = self.store.keys()
        self.table_stats = {k: TableStatistics(self.store, k)
                            for k in self.keys}

        # choose the smallest chunksize
        self.chunksize = min([self.table_stats[k].chunksize(self.allowed_mem)
                              for k in self.keys])

        # store for chunked data
        self.chunk_filename = os.path.splitext(filename)[0] + "_chunk.h5"
        self.chunk_store = pandas.HDFStore(self.chunk_filename, mode='w')

        # setup the output file
        self.shuffled_store = pandas.HDFStore(shuffled_filename, mode='w',
                                              complevel=self.complevel,
                                              complib=self.complib)


    def shuffle(self):
        """shuffle
           Shuffles all the tables in the HDFStore.

        """
        for k in self.keys:
            numpy.random.seed(self.seed)
            self.shuffle_table(k)

        self.store.close()
        self.shuffled_store.close()
        self.chunk_store.close()
        os.remove(self.chunk_filename)


    def shuffle_table(self, key):
        """shuffle_table
           Shuffle a table in the HDFStore, write to a new file.

        """
        # split up the table into chunks
        num_chunks, chunk_keys, chunk_counts = self.divide_table_into_chunks(key)

        # if there is one chunk, move it and finish
        if num_chunks == 1:
            self.shuffled_store.put(key, self.chunk_store[chunk_keys[0]],
                                    format='table')
            return

        self.reassemble_table(key, num_chunks, chunk_keys, chunk_counts)


    def divide_table_into_chunks(self, key):
        """divide_table_into_chunks
           Divides a table into chunks, each with their own table.
           Shuffles the chunked tables.

        """
        num_read = 0
        i_chunk = 0
        chunk_keys = []
        chunk_counts = []

        # get the column names
        column_names = list(self.store.select(key, start=0, stop=0))

        # read, shuffle, and write chunks
        while True:
            x = self.store.select(key, start=i_chunk*self.chunksize,
                                  stop=(i_chunk+1)*self.chunksize).as_matrix()
            numpy.random.shuffle(x)
            chunk_key = key + str(i_chunk)
            self.chunk_store.put(chunk_key,
                                 pandas.DataFrame(x, columns=column_names),
                                 format='table')

            # increment counters
            num_read += len(x)
            i_chunk += 1
            chunk_counts.append(len(x))
            chunk_keys.append(chunk_key)
            if num_read >= self.table_stats[key].shape[0]:
                break

        return (i_chunk, chunk_keys, chunk_counts)


    def reassemble_table(self, key, num_chunks, chunk_keys, chunk_counts):
        """reassemble_table
           Takes a set of chunked tables and rebuilds the shuffled table.

        """
        # find a streaming map
        stream_map = numpy.concatenate([chunk_counts[i]*[i]
                                        for i in range(len(chunk_counts))])
        numpy.random.shuffle(stream_map)

        # stream from the chunks into the shuffled store
        chunk_read_inds = num_chunks * [0]
        num_streamed = 0
        # read data in chunks
        for i_chunk in range(num_chunks):
            # get the count for each chunk table
            chunk_inds = stream_map[i_chunk*self.chunksize
                                    : (i_chunk+1)*self.chunksize]
            chunk_read_counts = [numpy.sum(chunk_inds == j)
                                 for j in range(num_chunks)]

            # now read chunks into an empty array
            arr = numpy.zeros((len(chunk_inds), self.table_stats[key].shape[1]),
                              self.table_stats[key].dtype)
            arr_ix = 0
            for j in range(num_chunks):
                num_read = chunk_read_counts[j]
                df_chunk = self.chunk_store.select(chunk_keys[j],
                                                   start=chunk_read_inds[j],
                                                   stop=chunk_read_inds[j] + num_read)
                arr[arr_ix : arr_ix + num_read] = df_chunk
                arr_ix += num_read
                chunk_read_inds[j] += num_read
            # shuffle the array and write it, setting the index
            numpy.random.shuffle(arr)
            df = pandas.DataFrame(arr, columns=list(df_chunk))
            df.index = range(num_streamed, num_streamed + len(arr))
            num_streamed += len(arr)
            self.shuffled_store.append(key, df)
