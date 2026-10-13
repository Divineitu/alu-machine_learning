#!/usr/bin/env python3
"""Performs PCA on a dataset and returns its transformed version."""

import numpy as np


def pca(X, ndim):
    """Return X projected onto its first ndim principal components."""
    X_m = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_m)
    return np.matmul(X_m, vh[:ndim].T)
