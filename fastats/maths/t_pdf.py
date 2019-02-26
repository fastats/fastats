
from numpy import power, sqrt, pi
from fastats.maths.gamma import gamma


def t_pdf(x, nu):
    """
    Student-t distribution Probability
    Density Function.

    PDF across `x` for a t distribution
    with degrees of freed `nu`.

    >>> t_pdf(0.56, 5)
    0.3162840947898818
    >>> t_pdf(-5, 2)
    0.007127781101036206
    >>> t_pdf(5, 2)
    0.007127781101036206
    """
    u = gamma(0.5 * (nu + 1))
    v1 = sqrt(nu * pi) * gamma(0.5 * nu)
    v2 = power(1 + power(x, 2) / nu, 0.5 * (nu + 1))
    out = u / (v1 * v2)
    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
