#!/usr/bin/env python3
"""Module for element-wise operations on numpy.ndarrays"""

def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication, and division
    Returns a tuple: (sum, difference, product, quotient)
    """
    import numpy as np
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
