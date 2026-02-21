# Probability

This project contains implementations of common probability distributions from scratch, without using external statistics libraries.

## Learning Objectives

- Understanding probability distributions (Poisson, Exponential, Normal, Binomial)
- Calculating probability mass functions (PMF) and probability density functions (PDF)
- Computing cumulative distribution functions (CDF)
- Working with distribution parameters and estimation from data

## Files

- `poisson.py` - Poisson distribution class with PMF and CDF methods
- `exponential.py` - Exponential distribution class with PDF and CDF methods
- `normal.py` - Normal (Gaussian) distribution class with z-scores, PDF, and CDF methods
- `binomial.py` - Binomial distribution class with PMF and CDF methods

## Requirements

- Python 3.5
- NumPy (for testing only, not used in implementations)
- pycodestyle 2.5

## Mathematical Constants

- π = 3.1415926536
- e = 2.7182818285

## Usage

```python
from poisson import Poisson
import numpy as np

# Create Poisson distribution from data
data = np.random.poisson(5., 100).tolist()
p = Poisson(data)
print(p.lambtha)  # Estimated lambda parameter
print(p.pmf(9))   # P(X = 9)
print(p.cdf(9))   # P(X <= 9)
```
