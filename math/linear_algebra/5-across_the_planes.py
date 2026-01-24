#!/usr/bin/env python3
"""Module for adding two 2D matrices element-wise"""

def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise

    Args:
        mat1: First 2D matrix (list of lists)
        mat2: Second 2D matrix (list of lists)

    Returns:
        A new 2D matrix with element-wise sums, or None if shapes differ
    """
    if len(mat1) != len(mat2):
        return None
    if any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None
    return [[row1[i] + row2[i] for i in range(len(row1))] for row1, row2 in zip(mat1, mat2)]
