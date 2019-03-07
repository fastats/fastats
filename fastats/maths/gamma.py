
import numpy as np
from numpy import log


GAMMA_COEFS = np.array([
    57.1562356658629235, -59.5979603554754912,
    14.1360979747417471, -0.491913816097620199,
    .339946499848118887e-4, .465236289270485756e-4,
    -.983744753048795646e-4, .158088703224912494e-3,
    -.210264441724104883e-3, .217439618115212643e-3,
    -.164318106536763890e-3, .844182239838527433e-4,
    -.261908384015814087e-4, .368991826595316234e-5
])


def gammaln(z):
    """
    Log Gamma function.

    Returns Log of the Gamma function for
    all `z` > 0; gammaln(z) = (z-1)!

    It is expected that all inputs
    are greater than zero.

    Given in Numerical Recipes 6.1


    >>> gammaln(4) # doctest: +ELLIPSIS
    1.79175946922805...
    >>> gammaln(11.23) # doctest: +ELLIPSIS
    15.64781466453367...
    >>> gammaln(85)
    291.32395009427034
    """
    assert np.greater(z, 0), "Values must be greater than zero!"
    y = z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * log(tmp) - tmp
    ser = np.ones_like(y) * 0.999999999999997092

    n = GAMMA_COEFS.shape[0]
    for j in range(n):
        y = y + 1
        ser = ser + GAMMA_COEFS[j] / y

    out = tmp + log(2.5066282746310005 * ser / z)
    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
