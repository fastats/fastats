
from numpy import power, exp
from fastats.maths.gamma import gammaln


def beta_pdf(x, alpha, beta):
    """
    Beta distribution Probability
    Density Function.

    PDF across `x` for a beta
    distribution with parameters
    `alpha` and `beta`.

    >>> beta_pdf(0.56, 1.2, 3.4) # doctest: +ELLIPSIS
    0.60688628807551...
    >>> beta_pdf(0.2, 2, 2) # doctest: +ELLIPSIS
    0.95999999999999...
    >>> beta_pdf(0.8, 2, 2) # doctest: +ELLIPSIS
    0.95999999999999...
    """
    u = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    u = exp(u)
    v = power(x, alpha - 1) * power(1 - x, beta - 1)
    out = u * v
    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
