#!/usr/bin/env python3
"""Calculates the precision for each class."""

import numpy as np


def precision(confusion):
    """Return the per-class precision from a confusion matrix."""
    return np.diag(confusion) / np.sum(confusion, axis=0)
