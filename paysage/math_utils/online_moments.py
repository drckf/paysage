"""
This module defines math utilities.

"""
import pandas

from paysage import backends as be

class MeanCalculator(object):
    """
    An online mean calculator.
    Calculates the mean of tensors, returning a single number.

    """
    def __init__(self):
        """
        Create a MeanCalculator object.

        Args:
            None

        Returns:
            The MeanCalculator object.

        """
        self.num = 0
        self.mean = 0

    def reset(self) -> None:
        """
        Resets the calculation to the initial state.

        Note:
            Modifies the metric in place.

        Args:
            None

        Returns:
            None

        """
        self.num = 0
        self.mean = 0

    def update(self, samples) -> None:
        """
        Update the online calculation of the mean.

        Notes:
            Modifies the metrics in place.

        Args:
            samples (tensor): data samples

        Returns:
            None

        """
        n = len(samples)
        sample_mean = be.mean(samples)

        # update the num and mean attributes
        self.num += n
        self.mean += (sample_mean - self.mean) * n / max(self.num, 1)


class MeanArrayCalculator(object):
    """
    An online mean calculator.
    Calculates the mean of a tensor along axes.
    Returns a tensor.

    """
    def __init__(self):
        """
        Create a MeanArrayCalculator object.

        Args:
            None

        Returns:
            The MeanArrayCalculator object.

        """
        self.num = None
        self.mean = None

    def reset(self) -> None:
        """
        Resets the calculation to the initial state.

        Note:
            Modifies the metric in place.

        Args:
            None

        Returns:
            None

        """
        self.num = None
        self.mean = None

    def update(self, samples, axis=0) -> None:
        """
        Update the online calculation of the mean.

        Notes:
            Modifies the metrics in place.

        Args:
            samples: data samples

        Returns:
            None

        """
        n = len(samples)
        sample_mean = be.mean(samples, axis=axis)

        # initialize the num and mean attributes if necessary
        if self.mean is None:
            self.mean = be.zeros_like(sample_mean)
            self.num = 0

        # update the num and mean attributes
        tmp = self.num*self.mean + n*sample_mean
        self.num += n
        self.mean = tmp / max(self.num, 1)
        #self.mean += (sample_mean - self.mean) * n / be.clip(self.num, a_min=1)


class MeanVarianceCalculator(object):
    """
    An online numerically stable mean and variance calculator.
    For computations on vector objects, where single values are returned.
    Uses Welford's algorithm for the variance.
    B.P. Welford, Technometrics 4(3):419–420.

    """
    def __init__(self):
        """
        Create MeanVarianceCalculator object.

        Args:
            None

        Returns:
            The MeanVarianceCalculator object.

        """
        self.num = 0
        self.mean = 0
        self.square = 0
        self.var = 0

    def reset(self) -> None:
        """
        Resets the calculation to the initial state.

        Note:
            Modifies the metrics in place.

        Args:
            None

        Returns:
            None

        """
        self.num = 0
        self.mean = 0
        self.square = 0
        self.var = 0

    def update(self, samples) -> None:
        """
        Update the online calculation of the mean and variance.

        Notes:
            Modifies the metrics in place.

        Args:
            samples: data samples

        Returns:
            None

        """
        n = len(samples)
        sample_mean = be.tsum(samples) / n
        sample_square = be.tsum(be.square(samples - sample_mean))

        delta = sample_mean - self.mean
        new_num = self.num + n
        correction = n*self.num*delta**2 / max(new_num, 1)

        self.square += sample_square + correction
        self.var = self.square / max(new_num-1, 1)
        self.mean = (self.num*self.mean + n*sample_mean) / max(new_num, 1)
        self.num = new_num


class MeanVarianceArrayCalculator(object):
    """
    An online numerically stable mean and variance calculator.
    For calculations on arrays, where tensor objects are returned.
    The variance over the 0-axis is computed.
    Uses Welford's algorithm for the variance.
    B.P. Welford, Technometrics 4(3):419–420.

    """
    def __init__(self):
        """
        Create MeanVarianceArrayCalculator object.

        Args:
            None

        Returns:
            The MeanVarianceArrayCalculator object.

        """
        self.num = None
        self.mean = None
        self.square = None
        self.var = None

    def reset(self) -> None:
        """
        Resets the calculation to the initial state.

        Note:
            Modifies the metrics in place.

        Args:
            None

        Returns:
            None

        """
        self.num = None
        self.mean = None
        self.square = None
        self.var = None

    def update(self, samples, axis=0) -> None:
        """
        Update the online calculation of the mean and variance.

        Notes:
            Modifies the metrics in place.

        Args:
            samples: data samples

        Returns:
            None

        """
        # compute the sample size and sample mean
        n = len(samples)
        sample_mean = be.tsum(samples, axis=axis) / max(n, 1)
        sample_square = be.tsum(be.square(be.subtract(sample_mean, samples)),
                                axis=axis)

        if self.mean is None:
            self.mean = be.zeros_like(sample_mean)
            self.square = be.zeros_like(sample_square)
            self.var = be.zeros_like(sample_square)
            self.num = 0


        delta = sample_mean - self.mean
        new_num = self.num + n
        correction = n*self.num*be.square(delta) / max(new_num, 1)

        self.square += sample_square + correction
        self.var = self.square / max(new_num-1, 1)
        self.mean = (self.num*self.mean + n*sample_mean) / max(new_num, 1)
        self.num = new_num

    @classmethod
    def from_dataframe(cls, df):
        """
        Create a MeanVarianceArrayCalculator from a DataFrame config.

        Args:
            config (DataFrame): the parameters, stored as a DataFrame.

        Returns:
            MeanVarianceArrayCalculator

        """
        mvac = cls()
        mvac.num = (df["num"].astype(int))[0] # constant column
        mvac.mean = be.float_tensor(df["mean"].astype(float))
        mvac.var = be.float_tensor(df["var"].astype(float))
        mvac.square = be.float_tensor(df["square"].astype(float))
        return mvac

    def to_dataframe(self):
        """
        Create a config DataFrame for the object.

        Args:
            None

        Returns:
            df (DataFrame): a DataFrame representation of the object.

        """
        if self.num is None:
            return pandas.DataFrame(None)

        df = pandas.DataFrame(None, index=range(len(self.mean)))
        # we have to store a whole column of self.num even though it is constant
        df["num"] = self.num * be.ones((len(self.mean),), dtype=be.Long)
        df["mean"] = be.to_numpy_array(self.mean)
        df["var"] = be.to_numpy_array(self.var)
        df["square"] = be.to_numpy_array(self.square)
        return df
