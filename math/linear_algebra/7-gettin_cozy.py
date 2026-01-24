#!/usr/bin/env python3
"""Module for concatenating two 2D matrices along a specific axis"""

def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along the specified axis

    Args:
        mat1: First 2D matrix (list of lists)
        mat2: Second 2D matrix (list of lists)
        axis: Axis along which to concatenate (0 for rows, 1 for columns)

    Returns:
        A new 2D matrix, or None if shapes are incompatible
    """
    if axis == 0:
        if len(mat1) == 0:
            return [row[:] for row in mat2]
        if len(mat2) == 0:
            return [row[:] for row in mat1]
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]
    else:
        return None
