#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian."""

import numpy as np


def pdf(X, m, S):
    """Evaluate a multivariate Gaussian PDF at each data point.

    Args:
        X: numpy.ndarray of shape (n, d) with the points to evaluate
        m: numpy.ndarray of shape (d,) with the mean of the distribution
        S: numpy.ndarray of shape (d, d) with the covariance

    Returns:
        P, a numpy.ndarray of shape (n,) with the PDF values,
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    d = X.shape[1]
    if not isinstance(m, np.ndarray) or m.shape != (d,):
        return None
    if not isinstance(S, np.ndarray) or S.shape != (d, d):
        return None

    diff = X - m
    mahal = np.sum(np.matmul(diff, np.linalg.inv(S)) * diff, axis=1)
    norm = np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S))
    P = np.exp(-mahal / 2) / norm
    return np.maximum(P, 1e-300)
