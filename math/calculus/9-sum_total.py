#!/usr/bin/env python3
"""Module for calculating summation of i squared."""


def summation_i_squared(n):
    """
    Calculate the sum of i squared from 1 to n.

    Args:
        n: The stopping condition (upper bound)

    Returns:
        The integer value of the sum, or None if n is not valid
    """
    if not isinstance(n, int) or n < 1:
        return None

    # Formula: sum(i^2) from 1 to n = n(n+1)(2n+1)/6
    return n * (n + 1) * (2 * n + 1) // 6
