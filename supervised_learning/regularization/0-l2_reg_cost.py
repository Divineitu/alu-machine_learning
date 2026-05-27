#!/usr/bin/env python3
"""Calculates the L2 regularized cost of a neural network."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Return the cost of the network with L2 regularization."""
    l2_sum = 0
    for i in range(1, L + 1):
        l2_sum += np.sum(weights['W' + str(i)] ** 2)
    return cost + (lambtha / (2 * m)) * l2_sum
