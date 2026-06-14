#!/usr/bin/env python3
"""Calculates the sensitivity (recall) for each class."""

import numpy as np


def sensitivity(confusion):
    """Return the per-class sensitivity from a confusion matrix."""
    return np.diag(confusion) / np.sum(confusion, axis=1)
