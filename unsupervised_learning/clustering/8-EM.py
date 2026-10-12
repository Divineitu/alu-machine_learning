#!/usr/bin/env python3
"""Performs the expectation maximization for a GMM."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5,
                             verbose=False):
    """Fit a GMM to X with the EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, the number of clusters
        iterations: positive integer, the maximum number of iterations
        tol: non-negative float, the log likelihood tolerance used for
            early stopping
        verbose: boolean, whether to print the log likelihood every
            10 iterations and after the last iteration

    Returns:
        pi, m, S, g, l, or None, None, None, None, None on failure
        pi: numpy.ndarray of shape (k,) with the priors
        m: numpy.ndarray of shape (k, d) with the centroid means
        S: numpy.ndarray of shape (k, d, d) with the covariances
        g: numpy.ndarray of shape (k, n) with the posteriors
        l: the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    msg = 'Log Likelihood after {} iterations: {}'
    pi, m, S = initialize(X, k)
    g, log_l = expectation(X, pi, m, S)
    prev_l = 0
    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(msg.format(i, round(log_l, 5)))
        pi, m, S = maximization(X, g)
        g, log_l = expectation(X, pi, m, S)
        if abs(log_l - prev_l) <= tol:
            break
        prev_l = log_l
    if verbose:
        print(msg.format(i + 1, round(log_l, 5)))
    return pi, m, S, g, log_l
