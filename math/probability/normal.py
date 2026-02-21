#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a Normal distribution."""

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal distribution.

        Args:
            data: List of data to estimate the distribution
            mean: Mean of the distribution
            stddev: Standard deviation of the distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate mean
            self.mean = float(sum(data) / len(data))
            # Calculate population standard deviation
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.

        Args:
            x: The x-value

        Returns:
            The z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value of a given z-score.

        Args:
            z: The z-score

        Returns:
            The x-value of z
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculate the PDF for a given x-value.

        Args:
            x: The x-value

        Returns:
            PDF value for x
        """
        # PDF = (1 / (σ * √(2π))) * e^(-((x-μ)²) / (2σ²))
        coefficient = 1 / (self.stddev * ((2 * Normal.pi) ** 0.5))
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        return coefficient * (Normal.e ** exponent)

    def cdf(self, x):
        """
        Calculate the CDF for a given x-value.

        Args:
            x: The x-value

        Returns:
            CDF value for x
        """
        # CDF = (1/2) * (1 + erf((x - μ) / (σ * √2)))
        # Using Taylor series approximation for erf
        value = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # erf approximation using Taylor series
        # erf(x) = (2/√π) * (x - x³/3 + x⁵/10 - x⁷/42 + x⁹/216 - ...)
        erf_approx = value - (value ** 3) / 3 + (value ** 5) / 10
        erf_approx = erf_approx - (value ** 7) / 42 + (value ** 9) / 216
        erf_approx *= 2 / (Normal.pi ** 0.5)

        cdf_value = (1 / 2) * (1 + erf_approx)
        return cdf_value
