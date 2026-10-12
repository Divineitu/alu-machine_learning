#!/usr/bin/env python3
"""Calculates a GMM from a dataset with scikit-learn."""

import sklearn.mixture


def gmm(X, k):
    """Fit a scikit-learn Gaussian Mixture Model to X.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: the number of clusters

    Returns:
        pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,) with the cluster priors
        m: numpy.ndarray of shape (k, d) with the centroid means
        S: numpy.ndarray of shape (k, d, d) with the covariances
        clss: numpy.ndarray of shape (n,) with the cluster indices
        bic: the BIC value of the fitted model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    clss = model.predict(X)
    return (model.weights_, model.means_, model.covariances_, clss,
            model.bic(X))
