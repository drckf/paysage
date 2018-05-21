import sys

from . import backends as be


class Transformation(object):

    def __init__(self, function=be.do_nothing, args=None, kwargs=None):
        """
        Create a transformation that operates on a list of tensors.

        Args:
            function (optional; callable)
            args (optional; List)
            kwargs (optional; Dict)

        Returns:
            Transformation

        """
        self.name = function.__name__
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def _closure(self):
        """
        Create a callable function with the arguments and keyword arguments
        already in place.

        Args:
            None

        Returns:
            callable

        """
        def partial(tensor):
            return self.function(tensor, *self.args, **self.kwargs)
        return partial

    def compute(self, tensor):
        """
        Apply the transformation to a single tensor.

        Args:
            tensor

        Returns:
            tensor

        """
        return self._closure()(tensor)

    def get_config(self):
        """
        Get the configuration of a transformation.

        Args:
            None

        Returns:
            Dict

        """
        return {'name': self.name,
                'args': self.args if len(self.args) > 0 else None,
                'kwargs': self.kwargs if len(self.kwargs) > 0 else None}

    @classmethod
    def from_config(cls, config):
        """
        Create a transformation from a configuration dictionary.

        Args:
            config (Dict)

        Returns:
            Transformation

        """
        function = getattr(sys.modules[__name__], config["name"])
        return cls(function, config['args'], config['kwargs'])


def scale(tensor, denominator=1):
    """
    Rescale the values in a tensor by the denominator.

    Args:
        tensor (tensor (num_samples, num_units))
        denominator (optional; float)

    Returns:
        tensor (tensor (num_samples, num_units))

    """
    return tensor/denominator


def l2_normalize(tensor):
    """
    Divide the rows of a tensor by their L2 norms.

    Args:
        tensor (tensor (num_samples, num_units))

    Returns:
        tensor (tensor (num_samples, num_units))

    """
    return be.divide(be.norm(tensor, axis=1, keepdims=True), tensor)


def l1_normalize(tensor):
    """
    Divide the rows of a tensor by their L1 norms.

    Args:
        tensor (tensor (num_samples, num_units))

    Returns:
        tensor (tensor (num_samples, num_units))

    """
    return be.divide(be.tsum(tensor, axis=1, keepdims=True), tensor)


def binarize_color(tensor):
    """
    Scales an int8 "color" value to [0, 1].

    Args:
        tensor (tensor (num_samples, num_units))

    Returns:
        tensor (tensor (num_samples, num_units))

    """
    return be.float_tensor(be.tround(tensor/255))


def one_hot(data, category_list):
    """
    Convert a categorical variable into a one-hot code.

    Args:
        data (tensor (num_samples, 1)): a column of the data matrix that is categorical
        category_list: the list of categories

    Returns:
        one-hot encoded data (tensor (num_samples, num_categories))

    """
    units = be.zeros((len(data), len(category_list)))
    on_units = be.long_tensor(list(map(category_list.index, be.flatten(data))))
    be.scatter_(units, on_units, 1.)
    return units
