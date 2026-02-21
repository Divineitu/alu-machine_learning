#!/usr/bin/env python3
"""Poisson distribution module."""


class Poisson:
    """Represents a Poisson distribution."""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson distribution.

        Args:
            data: List of data to estimate the distribution
            lambtha: Expected number of occurrences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculate the PMF for a given number of successes.

        Args:
            k: Number of successes

        Returns:
            PMF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        # Calculate k! (factorial)
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        # PMF = (λ^k * e^(-λ)) / k!
        pmf_value = ((self.lambtha ** k) *
                     (Poisson.e ** (-self.lambtha))) / factorial_k
        return pmf_value

    def cdf(self, k):
        """
        Calculate the CDF for a given number of successes.

        Args:
            k: Number of successes

        Returns:
            CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        # CDF = sum of PMF from 0 to k
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
