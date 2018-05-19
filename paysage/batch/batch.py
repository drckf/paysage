import numpy

from . import in_memory
from .. import preprocess as pre


def split_tensor(tensor, split_fraction):
    """
    Split a list of tensors into two parts into two fractions.
    Assumes the tensors are all the same length.

    Args:
        tensors (List[tensors]): the tensors to split.
        split_fraction (double): the fraction of the dataset to split at.

    Returns:
        tensors_part1 (tensor): the first part of the tensor.
        tensors_part2 (tensor): the second part of the tensor.

    """
    split = int(numpy.ceil(split_fraction * len(tensor)))
    return tensor[:split], tensor[split:]


def in_memory_batch(tensor, batch_size, train_fraction=0.9,
                    transform=pre.Transformation()):
    """
    Utility function to create a Batch object from a tensor.

    Args:
        tensor (tensors): the tensor to batch.
        batch_size (int): the (common) batch size.
        train_fraction (float): the fraction of data to use as training data.
        transform (callable): the (common) transform function.

    Returns:
        data (Batch): the batcher.

    """
    tensor_train, tensor_validate = split_tensor(tensor, train_fraction)
    return Batch({'train': in_memory.InMemoryTable(tensor_train, batch_size, transform),
                  'validate': in_memory.InMemoryTable(tensor_validate, batch_size, transform)})


class Batch(object):
    """
    Serves up minibatches from train and validation data sources.
    The data should probably be randomly shuffled
    if being used to train a model.

    """
    def __init__(self, batch_dictionary):
        """
        Holds data sources with iterators that can pull minibatches.
        The train and validate batchers must have methods:
            - set_parameters_with_test
            - get
            - get_by_index
            - reset_generator
            - close

        This object will hold the same dataset-level attributes
        as the training batcher.

        Args:
            batch_dictionary (Dict[str: InMemoryTable/HDFtable])

        Returns:
            A Batch instance.

        """
        self.batch = batch_dictionary
        self.modes = list(self.batch.keys())

        # get the key representing the training set
        if len(self.modes) == 1:
            self.train_key = self.modes[0]
        else:
            assert 'train' in self.modes, \
            "There needs to be a 'train' key if there is more than 1 table"
            self.train_key = 'train'

        # set dataset-level attributes
        self._set_dataset_attributes()

    def __enter__(self):
        """
        Trivial enter function for context managed use of Batch.
        """
        return self

    def __exit__(self, *args):
        """
        Exit function for context managed use of Batch.  Just calls close on all
        tables.
        """
        self.close()

    def _set_dataset_attributes(self):
        """
        Automatically set some attributes.

        Args:
            key (str)

        Returns:
            None

        """
        self.batch_size = self.batch[self.train_key].batch_size
        self.output_batch_size = self.batch[self.train_key].output_batch_size
        self.nrows = self.batch[self.train_key].nrows
        self.ncols = self.batch[self.train_key].ncols
        self.column_names = self.batch[self.train_key].column_names

    def get_transforms(self):
        """
        Return the transform functions.

        Args:
            None:

        Returns:
            Dict[Callable]

        """
        return {key: self.batch[key].transform for key in self.batch}

    def set_transforms(self, transforms):
        """
        Set the transform functions.

        Note:
            Modifes the batch[key].transforms attributes for key \in [train, validate]!

        Args:
            transforms (Dict[Callable])

        Returns:
            None

        """
        for key in transforms:
            self.batch[key].transform = transforms[key]

    def close(self, mode: str = 'all') -> None:
        """
        Close the data sources.

        Args:
            mode (str): the mode, 'train', 'validate', or 'all'.

        Returns:
            None

        """
        try:
            self.batch[mode].close()
        except KeyError:
            for mode in self.modes:
                self.batch[mode].close()

    def reset_generator(self, mode: str) -> None:
        """
        Reset the generator.

        Args:
            mode (str): the mode, 'train', 'validate', or 'all'.

        Returns:
            None

        """
        try:
            self.batch[mode].reset_generator()
        except KeyError:
            for mode in self.modes:
                self.batch[mode].reset_generator()

    def set_parameters_with_test(self, mode: str = 'all'):
        """
        Set the batch-dependent parameters with a test call to get.
        This allows to account for preprocess functions that transform
        the output batch size, number of steps, rows, or columns.

        Notes:
            Modifies batch attributes inplace.

        Args:
            mode (str): the mode, 'train', 'validate', or 'all'.

        Returns:
            None

        """
        try:
            self.batch[mode].set_parameters_with_test()
        except KeyError:
            for mode in self.modes:
                self.batch[mode].set_parameters_with_test()

        # reset the dataset-level attributes
        self._set_dataset_attributes()

    def get(self, mode: str):
        """
        Get the next minibatch.
        Will raise a StopIteration if the end of the data is reached.

        Args:
            mode (str): the mode to read, 'train' or 'validate'.

        Returns:
            tensor: the minibatch of data.

        """
        return self.batch[mode].get()

    def get_by_index(self, mode, index):
        """
        Get the next minibatch by index.

        Args:
            mode (str): the mode to read, 'train' or 'validate'.
            index (Listable): the index values to select.

        Returns:
            tensor: the minibatch of data.

        """
        return self.batch[mode].get_by_index(index)
