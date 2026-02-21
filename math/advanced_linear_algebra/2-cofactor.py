#!/usr/bin/env python3
"""Module for calculating the cofactor matrix of a matrix."""


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


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a matrix.

    Args:
        matrix: A list of lists whose cofactor matrix should be calculated

    Returns:
        The cofactor matrix of matrix

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

    # For 1x1 matrix, cofactor is [[1]]
    if n == 1:
        return [[1]]

    # Calculate cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
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

            # Cofactor is minor times (-1)^(i+j)
            minor_val = determinant(submatrix)
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_val)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
