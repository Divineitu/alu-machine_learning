#!/usr/bin/env python3
"""Module for transposing a 2D matrix"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix

    Args:
        matrix: A 2D list representing a matrix

    Returns:
        A new matrix that is the transpose of the input
    """
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]
