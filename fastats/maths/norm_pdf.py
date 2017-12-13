
from numpy import abs as npabs, sqrt, exp, pi


def norm_pdf(x, mu, sigma):
    """
    Normal distribution Probability
    Density Function.

    PDF across `x` for a normal
    distribution with mean `mu` and
    standard deviation `sigma`.

    >>> norm_pdf(0, 0.0, 1.0)
    0.3989422804014327
    >>> norm_pdf(-0.1, 0.0, 1.0)
    0.39695254747701181
    >>> norm_pdf(0.1, 0.0, 1.0)
    0.39695254747701181
    >>> norm_pdf(7, 5, 5) # doctest: +ELLIPSIS
    0.073654028060664...
    """
    a = npabs(sigma)
    u = (x - mu) / a
    v = sqrt(2 * pi) * a
    y = (1 / v) * exp(-u * u / 2)
    return y


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
