#!/usr/bin/env python3
"""Module for calculating the minor matrix of a matrix."""


def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Args:
        matrix: A list of lists whose determinant should be calculated

    Returns:
        The determinant of the matrix
    """
    n = len(matrix)

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: cofactor expansion along first row
    det = 0
    for j in range(n):
        # Create submatrix by removing first row and column j
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)

        # Add cofactor contribution
        sign = (-1) ** j
        det += sign * matrix[0][j] * determinant(submatrix)

    return det


def minor(matrix):
    """
    Calculate the minor matrix of a matrix.

    Args:
        matrix: A list of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is empty
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if all elements are lists
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Check for empty matrix [[]]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if matrix is square
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # For 1x1 matrix, minor is [[1]]
    if n == 1:
        return [[1]]

    # Calculate minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Create submatrix by removing row i and column j
            submatrix = []
            for row_idx in range(n):
                if row_idx != i:
                    sub_row = []
                    for col_idx in range(n):
                        if col_idx != j:
                            sub_row.append(matrix[row_idx][col_idx])
                    submatrix.append(sub_row)

            # Minor is the determinant of the submatrix
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix
