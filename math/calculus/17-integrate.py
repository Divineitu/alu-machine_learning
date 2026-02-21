#!/usr/bin/env python3
"""Module for calculating the integral of a polynomial."""


def poly_integral(poly, C=0):
    """
    Calculate the integral of a polynomial.

    Args:
        poly: A list of coefficients representing a polynomial.
              The index represents the power of x.
        C: The integration constant (default 0)

    Returns:
        A new list of coefficients representing the integral,
        or None if poly or C are not valid.
    """
    # Validate C
    if not isinstance(C, (int, float)):
        return None

    # Validate poly
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Check all coefficients are numbers
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Calculate integral: integral(a_n * x^n) = a_n * x^(n+1) / (n+1)
    integral = [C]
    for power, coef in enumerate(poly):
        new_coef = coef / (power + 1)
        # If coefficient is a whole number, represent as integer
        if new_coef == int(new_coef):
            new_coef = int(new_coef)
        integral.append(new_coef)

    # Remove trailing zeros to make list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
