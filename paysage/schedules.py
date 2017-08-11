from . import backends as be
import sys

def schedule_from_config(config):
    """
    Construct a schedule from a configuration.

    Args:
        A dictionary configuration of the metadata.

    Returns:
        Schedule

    """
    layer_obj = getattr(sys.modules[__name__], config[0])
    return layer_obj(**config[1])


class Schedule(object):
    """Base schedule class"""
    def get_config(self):
        return [self.__class__.__name__, self.__dict__]

    def copy(self):
        return schedule_from_config(self.get_config())


class Constant(Schedule):
    def __init__(self, initial=1.0, value=None):
        """
        Constant learning rate x(t) = x(0).

        Args:
            initial (float)

        Returns:
            Constant

        """
        self.initial = initial
        if value is None:
            self.reset()
        else:
            self.value = value

    def reset(self):
        self.value = self.initial

    def __next__(self):
        return be.float_scalar(self.value)


class ExponentialDecay(Schedule):
    def __init__(self, initial=1.0, coefficient=0.9, value=None):
        """
        Exponential decay with coefficient alpha, i.e. x(t) = alpha^t.
        Sets x(0) = 1 and uses the recursive formula x(t+1) = alpha * x(t).

        Args:
            initial (float)
            coefficient (float in [0,1])

        Returns:
            ExponentialDecay

        """
        self.initial = initial
        self.coefficient = coefficient
        if value is None:
            self.reset()
        else:
            self.value = value

    def reset(self):
        self.value = self.initial

    def __next__(self):
        self.value *= self.coefficient
        return be.float_scalar(self.value)


class PowerLawDecay(Schedule):
    def __init__(self, initial=1.0, coefficient=0.9, value=None):
        """
        Power law decay with coefficient alpha, i.e. x(t) = 1 / (1 + alpha * t).
        Sets x(0) = 1 and uses the recursive formula 1/x(t+1) = alpha + 1/x(t).

        Args:
            initial (float)
            coefficient (float in [0,1])

        Returns:
            PowerLawDecay

        """
        self.initial = initial
        self.coefficient = coefficient
        if value is None:
            self.reset()
        else:
            self.value = value

    def reset(self):
        self.value = self.initial

    def __next__(self):
        reciprocal = 1 / self.value
        reciprocal += self.coefficient
        self.value = 1 / reciprocal
        return be.float_scalar(self.value)
