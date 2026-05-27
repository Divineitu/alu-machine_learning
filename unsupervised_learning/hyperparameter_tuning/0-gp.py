#!/usr/bin/env python3
"""Represents a noiseless 1D Gaussian process."""

import numpy as np


class GaussianProcess:
    """Noiseless 1D Gaussian process with an RBF kernel."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the GP with the given samples and kernel parameters."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Return the RBF covariance matrix between X1 and X2."""
        sqdist = (X1 - X2.T) ** 2
        return self.sigma_f ** 2 * np.exp(-0.5 * sqdist / self.l ** 2)
