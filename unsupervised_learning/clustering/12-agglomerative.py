#!/usr/bin/env python3
"""Performs agglomerative clustering on a dataset."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Cluster X with Ward linkage and display the dendrogram.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: the maximum cophenetic distance for all clusters

    Returns:
        clss, a numpy.ndarray of shape (n,) with the cluster indices
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                            criterion='distance')
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
