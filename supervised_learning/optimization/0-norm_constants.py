#!/usr/bin/env python3
"""Calculates the normalization (standardization) constants of a matrix."""

import numpy as np


def normalization_constants(X):
    """Return the mean and standard deviation of each feature of X."""
    return np.mean(X, axis=0), np.std(X, axis=0)
