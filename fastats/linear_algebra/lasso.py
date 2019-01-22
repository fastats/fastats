
from numba import vectorize

from fastats.linear_algebra.ols import ols


def soft_threshold(x, threshold):
    """
    If abs(x) is less than or equal to threshold
    then return 0, else move the value towards
    0 by the threshold amount.
    """
    if x > threshold:
        val = x - threshold
    elif x < -threshold:
        val = x + threshold
    else:
        val = 0
    return val


SIGS = ['float64(float64, float64)', 'float32(float32, float32)']
soft_threshold_ = vectorize(SIGS)(soft_threshold)


def lasso_orthonormal(A, b, lambda_):
    """
    Lasso Regression coefficients estimation in
    the case that A is orthonormal.

    Replicates SKLearn Lasso behaviour for model
    where fit_intercept = False.

    lambda_ scaled by number of targets, per
    SKLearn.
    """
    lambda_ *= len(b)
    coefficients = ols(A, b)
    return soft_threshold_(coefficients, lambda_)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
