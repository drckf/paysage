import sys
from numpy import clip
from . import backends as be

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
        """
        Get a configuration dictionary for the schedule.

        Args:
            None

        Returns:
            dict

        """
        return [self.__class__.__name__, self.__dict__]

    def copy(self):
        """
        Copy a schedule.

        Args:
            None

        Returns:
            Schedule

        """
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
        """
        Reset the value of the schedule to the initial value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = self.initial

    def set_value(self, value):
        """
        Set the value of the schedule to the given value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = value

    def __next__(self):
        """
        Get the next value from the schedule.

        Args:
            None

        Returns:
            float

        """
        return be.float_scalar(self.value)


class Linear(Schedule):
    def __init__(self, initial=1.0, delta=0.0, value=None, minval=0.0, maxval=1.0):
        """
        Linear schedule x(t) = x(0) - delta t.

        Args:
            initial (float)
            delta (float)

        Returns:
            Linear

        """
        self.initial = initial
        self.delta = delta
        self.minval = minval
        self.maxval = maxval
        if value is None:
            self.reset()
        else:
            self.value = value

    def reset(self):
        """
        Reset the value of the schedule to the initial value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = self.initial

    def set_value(self, value):
        """
        Set the value of the schedule to the given value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = value

    def __next__(self):
        """
        Get the next value from the schedule.

        Args:
            None

        Returns:
            float

        """
        tmp = be.float_scalar(self.value)
        self.value -= self.delta
        self.value = clip(self.value, self.minval, self.maxval)
        return tmp


class Step(Schedule):
    def __init__(self, initial=1.0, final=0.0, steps=1, value=None):
        """
        Step function schedule:
            x(t) = initial if t < steps
            x(t) = final if t >= steps

        Args:
            initial (float)
            delta (float)

        Returns:
            Linear

        """
        self.initial = initial
        self.final = final
        self.steps = steps
        self.t = 0
        if value is None:
            self.reset()
        else:
            self.value = value

    def reset(self):
        """
        Reset the value of the schedule to the initial value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = self.initial
        self.t = 0

    def set_value(self, value):
        """
        Set the value of the schedule to the given value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = value

    def __next__(self):
        """
        Get the next value from the schedule.

        Args:
            None

        Returns:
            float

        """
        tmp = be.float_scalar(self.value)
        if self.t <= self.steps:
            self.value = self.initial
        else:
            self.value = self.final
        self.t += 1
        return tmp


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
        """
        Reset the value of the schedule to the initial value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = self.initial

    def set_value(self, value):
        """
        Set the value of the schedule to the given value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = value

    def __next__(self):
        """
        Get the next value from the schedule.

        Args:
            None

        Returns:
            float

        """
        tmp = be.float_scalar(self.value)
        self.value *= self.coefficient
        return tmp


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
        """
        Reset the value of the schedule to the initial value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = self.initial

    def set_value(self, value):
        """
        Set the value of the schedule to the given value.

        Notes:
            Modifies the value attribute in place!

        Args:
            None

        Returns:
            None

        """
        self.value = value

    def __next__(self):
        """
        Get the next value from the schedule.

        Args:
            None

        Returns:
            float

        """
        tmp = be.float_scalar(self.value)
        reciprocal = 1 / self.value
        reciprocal += self.coefficient
        self.value = 1 / reciprocal
        return tmp
