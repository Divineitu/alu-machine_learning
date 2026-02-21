# Calculus

This project covers fundamental calculus concepts including summation notation, derivatives, and integrals.

## Learning Objectives

- Understanding summation and product notation
- Computing derivatives using power rule, product rule, and chain rule
- Working with partial derivatives
- Calculating indefinite and definite integrals
- Evaluating double integrals

## Files

### Multiple Choice Questions (0-8, 11-16)
Answer files containing the correct option number for calculus problems involving:
- Sigma notation and summations
- Product notation
- Derivatives of polynomials and logarithms
- Partial derivatives
- Integrals and definite integrals
- Double integrals

### Python Functions
- `9-sum_total.py` - Calculate Σ(i²) from i=1 to n using the closed-form formula
- `10-matisse.py` - Calculate the derivative of a polynomial
- `17-integrate.py` - Calculate the integral of a polynomial

## Requirements

- Python 3.5
- No external libraries (numpy/scipy not allowed for calculations)
- pycodestyle 2.5

## Usage

```python
# Summation
summation_i_squared = __import__('9-sum_total').summation_i_squared
print(summation_i_squared(5))  # Output: 55

# Derivative
poly_derivative = __import__('10-matisse').poly_derivative
print(poly_derivative([5, 3, 0, 1]))  # Output: [3, 0, 3]

# Integral  
poly_integral = __import__('17-integrate').poly_integral
print(poly_integral([5, 3, 0, 1]))  # Output: [0, 5, 1.5, 0, 0.25]
```
