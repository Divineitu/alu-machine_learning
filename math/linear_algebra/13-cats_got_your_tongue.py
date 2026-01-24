#!/usr/bin/env python3
"""Module for concatenating two numpy.ndarrays along a specific axis"""

def np_cat(mat1, mat2, axis=0):
    """Concatenates two numpy.ndarrays along the specified axis"""
    import numpy as np
    return np.concatenate((mat1, mat2), axis=axis)
