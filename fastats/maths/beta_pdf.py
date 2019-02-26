
from numpy import power
from fastats.maths.gamma import gamma


def beta_pdf(x, alpha, beta):
    """
    Beta distribution Probability
    Density Function.

    PDF across `x` for a beta
    distribution with parameters
    `alpha` and `beta`.

    >>> beta_pdf(0.56, 1.2, 3.4) # doctest: +ELLIPSIS
    0.606886294679549...
    >>> beta_pdf(0.2, 2, 2)
    0.9600000335418806
    >>> beta_pdf(0.8, 2, 2)
    0.9600000335418805
    """
    u = gamma(alpha + beta) * power(1 - x, beta - 1) * power(x, alpha - 1)
    v = gamma(alpha) * gamma(beta)
    out = u / v
    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
