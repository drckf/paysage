import tempfile
import numpy
import pandas


class TableStatistics(object):
    """
    Stores basic statistics about a table.

    """
    def __init__(self, store, key):
        """
        Constructor.  Records the basic statistics.

        Args:
            store (HDFStore): the store with the table.
            key (str): the key of the table.

        Returns:
            A TableStatistics instance.

        """
        self.key_store = store.get_storer(key)

        self.shape = (self.key_store.nrows, self.key_store.ncols)
        self.dtype = self.key_store.dtype[1].base
        self.itemsize = self.dtype.itemsize
        self.mem_footprint = numpy.prod(self.shape) * self.itemsize / 1024**3 # in GiB

    def chunksize(self, allowed_mem):
        """
        Returns the sample count that will fit in allowed_mem,
        given the shape of the table.

        Args:
            allowed_mem (float): allowed memory in GiB

        Returns:
            chunksize (int): the chunk size that will fit in the allowed memory.

        """
        return int(self.shape[0] * allowed_mem / self.mem_footprint)


class DataShuffler(object):
    """
    Shuffles data in an HDF5 file.  Memory is managed.
    Synchronized shuffling between tables (with matching numbers of rows).

    Each table is shuffled with the following algorithm:
        - The table is sequentially divided into chunks.
            Each is shuffled and saved to a temporary file.
        - Pieces of each chunk are read in and shuffled together,
            Then written to the target file.
            This is done in a memory-managed way that preserves randomness.

    """
    def __init__(self, filename, shuffled_filename,
                 allowed_mem=1,
                 complevel=5,
                 seed=137):
        """
        Constructor.

        Args:
            filename (str): the filename of the data to be shuffled.
            shuffled_filename (str): the filename to write the shuffled data to.
            allowed_mem (float): the allowed memory footprint in GiB.
            complevel (int): the compression level used by pandas.
            seed (int): the random number seed for the shuffler.

        Returns:
            A DataShuffler instance.

        """
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
        chunk_tempfile = tempfile.NamedTemporaryFile()
        self.chunk_store = pandas.HDFStore(chunk_tempfile.name, mode='w')

        # setup the output file
        self.shuffled_store = pandas.HDFStore(shuffled_filename, mode='w',
                                              complevel=self.complevel,
                                              complib=self.complib)

    def shuffle(self):
        """
        Shuffles all the tables in the HDFStore.

        Args:
            None

        Returns:
            None

        """
        for k in self.keys:
            numpy.random.seed(self.seed)
            self.shuffle_table(k)

        self.store.close()
        self.shuffled_store.close()
        self.chunk_store.close()

    def shuffle_table(self, key):
        """
        Shuffle a table in the HDFStore, write to a new file.

        Args:
            key (str): the key of the table to shuffle.

        Returns:
            None

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
        """
        Divides a table into chunks, each with their own table.
        Shuffles the chunked tables.

        Args:
            key (str): the key of the table to chunk.

        Returns:
            num_chunks (int): the number of chunks the table was divided into.
            chunk_keys (List[str]): the keys of the chunks in the temporary file.
            chunk_counts (List[int]): the number of samples in each of the chunks.

        """
        num_read = 0
        i_chunk = 0
        chunk_keys = []
        chunk_counts = []

        # get the column names
        column_names = list(self.store.select(key, start=0, stop=0))

        # read, shuffle, and write chunks
        while num_read < self.table_stats[key].shape[0]:
            df_chunk = self.store.select(key, start=i_chunk*self.chunksize,
                                              stop=(i_chunk+1)*self.chunksize)
            chunk_key = key + str(i_chunk)
            self.chunk_store.put(chunk_key,
                                 df_chunk.sample(frac=1),
                                 format='table')

            # increment counters
            num_read += len(df_chunk)
            i_chunk += 1
            chunk_counts.append(len(df_chunk))
            chunk_keys.append(chunk_key)

        return (i_chunk, chunk_keys, chunk_counts)

    def reassemble_table(self, key, num_chunks, chunk_keys, chunk_counts):
        """
        Takes a set of chunked tables and rebuilds the shuffled table.

        Args:
            key (str): the key of the table to shuffle.
            num_chunks (int): the number of chunks the table was divided into.
            chunk_keys (List[str]): the keys of the chunks in the temporary file.
            chunk_counts (List[int]): the number of samples in each of the chunks.

        Returns:
            None

        """
        # find a streaming map
        stream_map = numpy.concatenate([[i for _ in range(chunk_counts[i])]
                                        for i in range(len(chunk_counts))])
        numpy.random.shuffle(stream_map)

        # stream from the chunks into the shuffled store
        chunk_read_inds = [0 for _ in range(num_chunks)]
        num_streamed = 0
        # read data in chunks
        for i_chunk in range(num_chunks):
            # get the number to read for each chunk table
            chunk_inds = stream_map[i_chunk*self.chunksize
                                    : (i_chunk+1)*self.chunksize]
            chunk_read_counts = [numpy.sum(chunk_inds == j)
                                 for j in range(num_chunks)]

            # now read the chunk pieces
            chunk_pieces = [None for _ in range(num_chunks)]
            for j in range(num_chunks):
                num_read = chunk_read_counts[j]
                chunk_pieces[j] = self.chunk_store.select(chunk_keys[j],
                                                   start=chunk_read_inds[j],
                                                   stop=chunk_read_inds[j] + num_read)
                chunk_read_inds[j] += num_read

            # combine the chunk pieces into a single chunk and shuffle
            df_chunk = pandas.concat(chunk_pieces)
            df_chunk = df_chunk.sample(frac=1)
            num_streamed += len(df_chunk)

            # write the chunk
            self.shuffled_store.append(key, df_chunk)
