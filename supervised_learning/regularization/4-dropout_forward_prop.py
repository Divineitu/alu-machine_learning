#!/usr/bin/env python3
"""Conducts forward propagation using Dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Return a cache of activations and dropout masks for each layer."""
    cache = {'A0': X}
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        z = np.matmul(W, cache['A' + str(i - 1)]) + b
        if i == L:
            t = np.exp(z - np.max(z, axis=0, keepdims=True))
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            a = np.tanh(z)
            d = np.random.binomial(1, keep_prob, size=a.shape)
            cache['A' + str(i)] = a * d / keep_prob
            cache['D' + str(i)] = d
    return cache
