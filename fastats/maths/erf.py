
from numpy import exp


def erf(x):
    """
    Error Function

    Abramowitz + Stegun
    Abramowitz + Stegun
    Handbook of Mathematical Functions,
    formula 7.1.26, and johndcook.com

    >>> erf(0.5)  # doctest: +ELLIPSIS
    0.52050001...
    >>> erf(1.0)  # doctest: +ELLIPSIS
    0.84270068...
    >>> erf(-0.5)  # doctest: +ELLIPSIS
    0.52089577...
    """
    t = 1.0 / (1.0 + 0.3275911 * x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    y = 1.0 - (((((a5*t + a4)*t) + a3)*t
                + a2)*t + a1)*t*exp(-x*x)
    return sign*y


if __name__ == '__main__':
    import pytest
    pytest.main()
