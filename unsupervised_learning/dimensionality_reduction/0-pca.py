#!/usr/bin/env python3
"""Performs PCA on a dataset."""

import numpy as np


def pca(X, var=0.95):
    """Return the weights matrix that maintains var fraction of variance."""
    _, s, vh = np.linalg.svd(X)
    ratio = np.cumsum(s) / np.sum(s)
    nd = np.argmax(ratio >= var) + 1
    return vh[:nd].T
