#!/usr/bin/env python3
"""Module for calculating a correlation matrix."""
import numpy as np


def correlation(C):
    """
    Calculate a correlation matrix from a covariance matrix.

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix

    Raises:
        TypeError: If C is not a numpy.ndarray
        ValueError: If C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Correlation: C_ij / sqrt(C_ii * C_jj)
    stddevs = np.sqrt(np.diag(C))
    outer = np.outer(stddevs, stddevs)
    correlation_matrix = C / outer

    return correlation_matrix
