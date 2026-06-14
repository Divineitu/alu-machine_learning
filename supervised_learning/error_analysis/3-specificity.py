#!/usr/bin/env python3
"""Calculates the specificity for each class."""

import numpy as np


def specificity(confusion):
    """Return the per-class specificity from a confusion matrix."""
    total = np.sum(confusion)
    actual_pos = np.sum(confusion, axis=1)
    predicted_pos = np.sum(confusion, axis=0)
    true_pos = np.diag(confusion)
    false_pos = predicted_pos - true_pos
    true_neg = total - actual_pos - false_pos
    return true_neg / (true_neg + false_pos)
