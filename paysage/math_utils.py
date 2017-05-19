# -*- coding: utf-8 -*-
"""
This module defines math utilities.

"""

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
            Modifies the metric in place.

        Args:
            None

        Returns:
            None

        """
        self.num = 0
        self.mean = 0
        self.var = 0

    def calculate(self, samples):
        """
        Run an online calculation of the mean and variance.

        Notes:
            The unnormalized variance is calculated
                (not divided by the number of samples).

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
