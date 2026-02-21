#!/usr/bin/env python3
"""Binomial distribution module."""


class Binomial:
    """Represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize Binomial distribution.

        Args:
            data: List of data to estimate the distribution
            n: Number of Bernoulli trials
            p: Probability of success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean and variance from data
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # For binomial: mean = np, variance = np(1-p)
            # So: variance/mean = 1 - p => p = 1 - variance/mean
            # Then: n = mean / p
            p_estimate = 1 - (variance / mean)
            n_estimate = mean / p_estimate

            # Round n to nearest integer
            self.n = round(n_estimate)

            # Recalculate p with rounded n
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculate the PMF for a given number of successes.

        Args:
            k: Number of successes

        Returns:
            PMF value for k
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        # PMF = C(n,k) * p^k * (1-p)^(n-k)
        # C(n,k) = n! / (k! * (n-k)!)
        coefficient = self._factorial(self.n) / (
            self._factorial(k) * self._factorial(self.n - k)
        )
        pmf_value = coefficient * (self.p ** k) * ((1 - self.p) ** (self.n - k))
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

    def _factorial(self, n):
        """
        Calculate factorial of n.

        Args:
            n: Non-negative integer

        Returns:
            n!
        """
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
