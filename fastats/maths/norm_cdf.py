
from numpy import sqrt, minimum

from fastats.maths.erfc import erfc


def norm_cdf(x, mu, sigma):
    """
    Normal Cumulative Distribution
    Function

    Norm CDF across X for a normal
    distribution with mean `mu` and
    standard deviation `sigma`.

    >>> norm_cdf(0.0, 0.0, 1.0) # doctest: +ELLIPSIS
    0.5000000...
    >>> norm_cdf(0.1, 0.0, 1.0) # doctest: +ELLIPSIS
    0.53982786...
    >>> norm_cdf(-0.1, 0.0, 1.0) # doctest: +ELLIPSIS
    0.460172135...
    >>> norm_cdf(90, 100, 4) # doctest: +ELLIPSIS
    0.00620966...
    """
    t = x - mu
    s = sigma * sqrt(2.0)
    y = 0.5 * erfc(-t / s)
    return minimum(y, 1.0)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
