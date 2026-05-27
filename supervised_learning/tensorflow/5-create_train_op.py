#!/usr/bin/env python3
"""Creates the training operation for a neural network."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Return the gradient descent training operation."""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
