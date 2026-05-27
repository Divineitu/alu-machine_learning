#!/usr/bin/env python3
"""Normalizes (standardizes) a matrix using precomputed constants."""


def normalize(X, m, s):
    """Return the matrix X normalized to zero mean and unit variance."""
    return (X - m) / s
