import numpy
import pandas

from .. import backends as be
from .. import preprocess as pre

def maybe_int(x):
    try:
        return int(x)
    except ValueError:
        return x

class HDFtable(object):
    """
    Serves up minibatches from a single table in an HDFStore.
    The data should probably be randomly shuffled
    if being used to train a model.

    """
    def __init__(self, filename, key, batch_size, transform=pre.Transformation(),
                 combine_frames=False):
        """
        Creates an iterator that can pull minibatches from an HDFStore.
        Works on a single table.

        Args:
            filename (str): the HDFStore file to read from.
            key (str): the key of the table to read from.
            batch_size (int): the minibatch size.
            transform (Transformation): the transform function to apply to the data.
            combine_frames (optional; bool): datasets with too many columns
                have to be divided into chunks. These chunks are stored as
                frames in the hdf5 file.

        Returns:
            An HDFtable instance.

        """
        self.transform = transform
        self.combine_frames = combine_frames

        # open the store, get the dimensions of the keyed table
        self.store = pandas.HDFStore(filename, mode='r')
        self.key = key
        self.batch_size = batch_size
        self.output_batch_size = batch_size

        self.ncols = self.store.get_storer(key).ncols
        self.nrows = self.store.get_storer(key).nrows

        self.column_names = self._get_column_names()

        # create and iterator over the data
        self.iterator = self.store.select(self.key, iterator=True,
                                          chunksize=self.batch_size)
        self.generator = self.iterator.__iter__()

        # change parameters as needed with a test call
        self.set_parameters_with_test()

    def _get_column_names(self):
        cols = self.store.select(self.key, start=0, stop=0).columns
        return list(cols.get_level_values(cols.nlevels-1))

    def close(self) -> None:
        """
        Close the HDFStore.

        Args:
            None

        Returns:
            None

        """
        self.store.close()

    def reset_generator(self) -> None:
        """
        Reset the generator.

        Args:
            mode (str): the mode, 'train', 'validate', or 'all'.

        Returns:
            None

        """
        self.generator = self.iterator.__iter__()

    def set_parameters_with_test(self):
        """
        Set the batch-dependent parameters with a test call to get.
        This allows to account for preprocess functions that transform
        the output batch size, rows, or columns.

        Notes:
            Modifies output_batch_size attribute in place, resets the generator.

        Args:
            None

        Returns:
            None

        """
        self.output_batch_size = len(self.get())
        self.reset_generator()

    def get(self):
        """
        Get the next minibatch.
        Will raise a StopIteration if the end of the data is reached.

        Args:
            None

        Returns:
            tensor: the minibatch of data.

        """
        try:
            vals = be.float_tensor(numpy.array(next(self.generator)))
        except StopIteration:
            self.reset_generator()
            raise StopIteration
        trans_vals = self.transform.compute(vals)
        return trans_vals

    def get_by_index(self, index):
        """
        Get the next minibatch by index.

        Args:
            index (Listable): the index values to select.

        Returns:
            tensor: the minibatch of data.

        """
        idx = list(index)
        vals = self.store.select(self.key, where="index={}".format(idx))
        return self.transform.compute(be.float_tensor(numpy.array(vals)))
