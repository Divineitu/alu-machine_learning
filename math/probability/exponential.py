#!/usr/bin/env python3
"""Exponential distribution module."""


class Exponential:
    """Represents an Exponential distribution."""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential distribution.

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
            # For exponential distribution, mean = 1/lambda
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculate the PDF for a given time period.

        Args:
            x: Time period

        Returns:
            PDF value for x
        """
        if x < 0:
            return 0

        # PDF = λ * e^(-λx)
        pdf_value = self.lambtha * (Exponential.e ** (-self.lambtha * x))
        return pdf_value

    def cdf(self, x):
        """
        Calculate the CDF for a given time period.

        Args:
            x: Time period

        Returns:
            CDF value for x
        """
        if x < 0:
            return 0

        # CDF = 1 - e^(-λx)
        cdf_value = 1 - (Exponential.e ** (-self.lambtha * x))
        return cdf_value
