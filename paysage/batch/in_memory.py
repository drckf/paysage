from .. import backends as be
from .. import preprocess as pre

def inclusive_slice(tensor, start, stop, step):
    """
    Generator yielding progressive inclusive slices from a tensor.

    Args:
        tensor (tensors): the tensors to minibatch.
        start (int): the start index.
        stop (int): the stop index.
        step (int): the minibatch size.

    Returns:
        tensor (tensor): a minibatch of tensors.

    """
    current = start
    while current < stop:
        next_iter = min(stop, current + step)
        result = (current, next_iter)
        current = next_iter
        yield tensor[result[0]:result[1]]


class InMemoryTable(object):
    """
    Serves up minibatches from a tensor held in memory.
    The data should probably be randomly shuffled
    if being used to train a model.

    """
    def __init__(self, tensor, batch_size, transform=pre.Transformation()):
        """
        Creates iterators that can pull minibatches
        from a list of in-memory arrays.

        Args:
            tensor (tensors): the array to batch
            batch_size (int): the minibatch size
            transform (Transformation): the transform function to apply to the data

        Returns:
            An InMemoryTable instance.

        """
        self.tensor = tensor
        self.batch_size = batch_size
        self.output_batch_size = batch_size
        self.transform = transform
        self.nrows, self.ncols = be.shape(self.tensor)
        self.column_names = list(range(self.ncols))

        # create iterators over the data for the train/validate sets
        self.iterators = inclusive_slice(self.tensor, 0, self.nrows,
                                         self.batch_size)

        # change parameters as needed with a test call
        self.set_parameters_with_test()

    def close(self) -> None:
        """
        Frees the tensor.
        """
        del self.tensor

    def reset_generator(self) -> None:
        """
        Reset the generator.

        Args:
            None

        Returns:
            None

        """
        self.iterators = inclusive_slice(self.tensor, 0, self.nrows,
                                         self.batch_size)

    def set_parameters_with_test(self):
        """
        Set the batch-dependent parameters with a test call to get.
        This allows to account for preprocess functions that transform
        the output batch size, number of steps, rows, or columns.

        Notes:
            Modifies output_batch_size.

        Args:
            None

        Returns:
            None

        """
        self.output_batch_size = be.shape(self.get())[0]
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
            vals = next(self.iterators)
        except StopIteration:
            self.reset_generator()
            raise StopIteration
        trans_vals = self.transform.compute(vals)
        return trans_vals

    def get_by_index(self, index):
        """
        Get the next minibatch by index.

        Args:
            index (tensor): the index values to select.

        Returns:
            tensor: the minibatch of data.

        """
        return self.transform.compute(self.tensor[index])
