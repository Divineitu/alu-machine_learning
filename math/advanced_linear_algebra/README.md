# Advanced Linear Algebra

This project contains implementations of fundamental linear algebra operations without using external libraries (except NumPy for the definiteness function).

## Learning Objectives

- Understanding determinants and how to calculate them
- Working with minors, cofactors, and adjugate matrices
- Computing matrix inverses
- Determining matrix definiteness using eigenvalues

## Files

- `0-determinant.py` - Calculate the determinant of a matrix
- `1-minor.py` - Calculate the minor matrix of a matrix
- `2-cofactor.py` - Calculate the cofactor matrix of a matrix
- `3-adjugate.py` - Calculate the adjugate matrix of a matrix
- `4-inverse.py` - Calculate the inverse of a matrix
- `5-definiteness.py` - Determine if a matrix is positive definite, negative definite, etc.

## Requirements

- Python 3.5
- NumPy 1.15 (only for definiteness function)
- pycodestyle 2.5

## Usage

Each module can be imported and used independently:

```python
determinant = __import__('0-determinant').determinant
matrix = [[1, 2], [3, 4]]
print(determinant(matrix))  # Output: -2
```
