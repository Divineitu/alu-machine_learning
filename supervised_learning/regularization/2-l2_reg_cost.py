#!/usr/bin/env python3
"""Calculates the L2 regularized cost of a tensorflow neural network."""

import tensorflow as tf


def l2_reg_cost(cost):
    """Return the network cost combined with the L2 regularization losses."""
    return cost + tf.losses.get_regularization_losses()
