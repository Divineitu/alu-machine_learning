#!/usr/bin/env python3
"""Creates a confusion matrix from one-hot labels and predictions."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Return the confusion matrix of correct labels vs predictions."""
    return np.matmul(labels.T, logits)
