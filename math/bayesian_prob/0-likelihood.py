#!/usr/bin/env python3
"""Module for calculating likelihood of data given probabilities."""
import numpy as np
from math import factorial


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining data given hypothetical
    probabilities.

    Args:
        x: Number of patients that develop severe side effects
        n: Total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities

    Returns:
        1D numpy.ndarray containing the likelihood for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient: n! / (x! * (n-x)!)
    # Use log-space to avoid overflow
    binom_coeff = factorial(n) / (factorial(x) * factorial(n - x))

    # Likelihood = C(n,x) * p^x * (1-p)^(n-x)
    L = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return L
