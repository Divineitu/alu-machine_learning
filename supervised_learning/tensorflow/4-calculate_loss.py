#!/usr/bin/env python3
"""Calculates the softmax cross-entropy loss of a prediction."""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Return the softmax cross-entropy loss tensor."""
    return tf.losses.softmax_cross_entropy(y, y_pred)
