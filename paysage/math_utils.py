# -*- coding: utf-8 -*-
"""
This module defines math utilities.

"""

from paysage import backends as be

class MeanCalculator(object):
    """
    An online mean calculator.

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

    def update(self, samples, **kwargs) -> None:
        """
        Update the online calculation of the mean.

        Notes:
            Modifies the metrics in place.

        Args:
            samples: data samples

        Returns:
            None

        """
        num_samples = len(samples)
        self.num += num_samples
        self.mean = self.mean + (be.mean(samples, **kwargs) - self.mean) * \
                                num_samples / self.num


class MeanVarianceCalculator(object):
    """
    An online numerically stable mean and variance calculator.
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
        for s in samples:
            self.num += 1
            mean_update = self.mean + (s - self.mean) / self.num
            self.var = (self.var*(self.num - 1) + (s - mean_update)*(s - self.mean)) / self.num
            self.mean = mean_update
