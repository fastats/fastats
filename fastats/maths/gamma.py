
import numpy as np
from numpy import log, exp, pi, power


SMALL_GAMMA_COEFS = np.array([
    1., 0.5772156649015329, -0.6558780715202538, -0.0420026350340952, 0.1665386113822915, -0.0421977345555443,
    -0.0096219715278770, 0.0072189432466630, -0.0011651675918591, -0.0002152416741149, 0.0001280502823882,
    -0.0000201348547807, -0.0000012504934821, 0.0000011330272320, -0.0000002056338417, 0.0000000061160950,
    0.0000000050020075, -0.0000000011812746, 0.0000000001043427, 0.0000000000077823, -0.0000000000036968,
    0.0000000000005100, -0.0000000000000206, -0.0000000000000054, 0.0000000000000014, 0.0000000000000001
])


def _gammaln_weier(z):
    """
    Log Gamma function.
    
    Uses the Weierstrass series. Specifically given in Abramowitz 6.1.41 (Tenth printing)
    
    >>> _gammaln_weier(8.01) # doctest: +ELLIPSIS
    8.54532442974...
    >>> _gammaln_weier(4.71) # doctest: +ELLIPSIS
    2.75079122402...
    >>> _gammaln_weier(55.001) # doctest: +ELLIPSIS
    164.32411048709...
    """
    assert z > 0.

    u = (z - 0.5) * log(z) - z + 0.5 * log(2 * pi)
    v = 1 / (12 * z) - 1 / (360 * power(z, 3)) + 1 / (1260 * power(z, 5)) - 1 / (1680 * power(z, 7))
    out = u + v
    return out


def _reciprocal_gamma(z):
    """
    Reciprocal of the Gamma function.
    
    Uses Abramowitz series expansion 6.1.34 (Tenth printing)
    
    Works very well for z < 3.
    
    >>> _reciprocal_gamma(2) # doctest: +ELLIPSIS
    1.00000001888...
    >>> _reciprocal_gamma(1.414) # doctest: +ELLIPSIS
    1.12791665430...
    >>> _reciprocal_gamma(0.0731) # doctest: +ELLIPSIS
    0.07592735941...
    """
    assert z > 0.

    n = SMALL_GAMMA_COEFS.shape[0]
    out = 0
    for k in range(1, n + 1):
        out += SMALL_GAMMA_COEFS[k - 1] * power(z, k)
    return out


def gamma(z):
    """
    Gamma function.
    
    This function uses the two gamma functions with minimum
    error, which will be demonstrated below.

    >>> gamma(7.1) # doctest: +ELLIPSIS
    868.95685878536...
    >>> gamma(0.99) # doctest: +ELLIPSIS
    1.0058719796441...
    >>> gamma(13.8) # doctest: +ELLIPSIS
    3705442052.278129...
    """
    assert z > 0.

    threshold_ = 2.24  # this is the _reciprocal_gamma + _gammaln_weier accuracy cross-over
    if z < threshold_:
        return 1 / _reciprocal_gamma(z)
    else:
        return exp(_gammaln_weier(z))


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
