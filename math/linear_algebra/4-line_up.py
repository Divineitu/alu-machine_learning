#!/usr/bin/env python3
"""Module for adding two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise

    Args:
        arr1: First list of ints/floats
        arr2: Second list of ints/floats

    Returns:
        A new list with element-wise sums, or None if shapes differ
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
