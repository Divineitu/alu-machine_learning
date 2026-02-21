#!/usr/bin/env python3
"""Module for calculating the derivative of a polynomial."""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly: A list of coefficients representing a polynomial.
              The index represents the power of x.

    Returns:
        A new list of coefficients representing the derivative,
        or None if poly is not valid.
    """
    # Validate poly
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check all coefficients are numbers
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # If polynomial is just a constant, derivative is 0
    if len(poly) == 1:
        return [0]

    # Calculate derivative: d/dx(a_n * x^n) = n * a_n * x^(n-1)
    derivative = []
    for power in range(1, len(poly)):
        derivative.append(power * poly[power])

    # If all coefficients are 0, return [0]
    if all(c == 0 for c in derivative):
        return [0]

    return derivative
