#!/usr/bin/env python3
"""Bayesian optimization with the Expected Improvement acquisition."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the optimizer with the black-box function and prior."""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], ac_samples
        ).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Return the next best sample and the EI values over X_s."""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            f_best = np.min(self.gp.Y)
            imp = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            imp = mu - f_best - self.xsi

        Z = np.zeros_like(sigma)
        mask = sigma != 0
        Z[mask] = imp[mask] / sigma[mask]
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[~mask] = 0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
