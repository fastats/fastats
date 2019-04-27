
from numpy import exp, power, sqrt, pi
from fastats.maths.gamma import gammaln


def t_pdf(x, nu):
    """
    Student-t distribution Probability
    Density Function.

    PDF across `x` for a t distribution
    with degrees of freed `nu`.

    >>> t_pdf(0.56, 5) # doctest: +ELLIPSIS
    0.316284053187681...
    >>> t_pdf(-5, 2) # doctest: +ELLIPSIS
    0.007127781101106...
    >>> t_pdf(5, 2) # doctest: +ELLIPSIS
    0.007127781101106...
    """
    u = gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)
    u = exp(u)
    v = sqrt(nu * pi) * power(1 + power(x, 2) / nu, 0.5 * (nu + 1))
    out = u / v
    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
