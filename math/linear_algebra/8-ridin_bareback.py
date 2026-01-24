#!/usr/bin/env python3
"""Module for matrix multiplication of two 2D matrices"""

def mat_mul(mat1, mat2):
    """Performs matrix multiplication on two 2D matrices

    Args:
        mat1: First 2D matrix (list of lists)
        mat2: Second 2D matrix (list of lists)

    Returns:
        A new 2D matrix that is the product, or None if shapes are incompatible
    """
    if len(mat1) == 0 or len(mat2) == 0 or len(mat1[0]) != len(mat2):
        return None
    result = []
    for row in mat1:
        new_row = []
        for col in range(len(mat2[0])):
            val = sum(row[k] * mat2[k][col] for k in range(len(mat2)))
            new_row.append(val)
        result.append(new_row)
    return result
