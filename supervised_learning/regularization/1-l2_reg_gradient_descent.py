#!/usr/bin/env python3
"""Updates weights and biases using gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update the network's weights and biases in place using L2 reg GD."""
    m = Y.shape[1]
    weights_copy = weights.copy()
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]
        b = weights_copy['b' + str(i)]
        dw = np.matmul(dz, A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dz, axis=1, keepdims=True) / m
        if i > 1:
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)
        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = b - alpha * db
