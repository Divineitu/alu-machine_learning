#!/usr/bin/env python3
"""Creates the momentum training operation in tensorflow."""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Return the gradient descent with momentum optimization op."""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
