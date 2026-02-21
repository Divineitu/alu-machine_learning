#!/usr/bin/env python3
"""Module for calculating the definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """
    Calculate the definiteness of a matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n) whose definiteness
                should be calculated

    Returns:
        The string 'Positive definite', 'Positive semi-definite',
        'Negative semi-definite', 'Negative definite', or 'Indefinite'
        if the matrix is positive definite, positive semi-definite,
        negative semi-definite, negative definite or indefinite,
        respectively. Returns None if matrix does not fit any category.

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is valid (non-empty and 2D square)
    if matrix.ndim != 2:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.shape[0] == 0:
        return None

    # Check if matrix is symmetric (required for definiteness)
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check definiteness based on eigenvalues
    all_positive = np.all(eigenvalues > 0)
    all_non_negative = np.all(eigenvalues >= 0)
    all_negative = np.all(eigenvalues < 0)
    all_non_positive = np.all(eigenvalues <= 0)

    if all_positive:
        return "Positive definite"
    elif all_non_negative:
        return "Positive semi-definite"
    elif all_negative:
        return "Negative definite"
    elif all_non_positive:
        return "Negative semi-definite"
    else:
        return "Indefinite"
