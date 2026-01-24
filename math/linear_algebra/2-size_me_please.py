#!/usr/bin/env python3
"""Module for calculating the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix

    Args:
        matrix: A nested list representing a matrix

    Returns:
        A list of integers representing the shape
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
