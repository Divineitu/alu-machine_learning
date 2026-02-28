#!/usr/bin/env python3
"""Module for Multivariate Normal distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Initialize MultiNormal distribution.

        Args:
            data: numpy.ndarray of shape (d, n) containing the data set

        Raises:
            TypeError: If data is not a 2D numpy.ndarray
            ValueError: If data contains less than 2 data points
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean: shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Covariance: (1/(n-1)) * (data - mean) @ (data - mean)^T
        X_centered = data - self.mean
        self.cov = (X_centered @ X_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculate the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1) containing the data point

        Returns:
            The value of the PDF

        Raises:
            TypeError: If x is not a numpy.ndarray
            ValueError: If x does not have the shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # PDF = 1 / (sqrt((2π)^d * det(Σ))) * exp(-0.5 * (x-μ)^T Σ^-1 (x-μ))
        diff = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)

        exponent = -0.5 * (diff.T @ cov_inv @ diff)
        coefficient = 1.0 / np.sqrt(((2 * np.pi) ** d) * cov_det)

        pdf_value = float(coefficient * np.exp(exponent))

        return pdf_value
