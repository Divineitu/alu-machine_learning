#!/usr/bin/env python3
"""Gaussian process with prediction and update."""

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

    def predict(self, X_s):
        """Return the posterior mean and variance of the GP at X_s."""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)
        cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov)
        return mu, sigma

    def update(self, X_new, Y_new):
        """Append a new sample to the GP and refresh the kernel matrix."""
        self.X = np.append(self.X, X_new.reshape(-1, 1), axis=0)
        self.Y = np.append(self.Y, Y_new.reshape(-1, 1), axis=0)
        self.K = self.kernel(self.X, self.X)
