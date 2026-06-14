# Error Analysis

NumPy implementations of the core classifier evaluation metrics computed
from a multiclass confusion matrix, plus two short written tasks on
diagnosing bias and variance from training and validation results.

## Files

- `0-create_confusion.py` — builds a confusion matrix from one-hot label
  and prediction matrices.
- `1-sensitivity.py` — per-class sensitivity (recall): diagonal divided by
  the actual-positive row sums.
- `2-precision.py` — per-class precision: diagonal divided by the
  predicted-positive column sums.
- `3-specificity.py` — per-class specificity: true negatives over
  actual negatives.
- `4-f1_score.py` — per-class F1 score, computed from sensitivity and
  precision.
- `5-error_handling` — answers for the bias/variance scenario question.
- `6-compare_and_contrast` — answer for the train-vs-validation
  confusion-matrix comparison question.

## Requirements

- Python 3
- NumPy

## Data

The example main scripts use `labels_logits.npz` and `confusion.npz`. These
files are not committed to this repository.
