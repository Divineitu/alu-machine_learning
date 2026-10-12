#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize the priors, means and covariances of a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters

    Returns:
        pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,) with evenly initialized priors
        m: numpy.ndarray of shape (k, d) with means from K-means
        S: numpy.ndarray of shape (k, d, d) of identity covariances
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None
    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
