
import numpy as np
from numpy import exp


def erf(x):
    """
    Error Function
    
    Returns error function for an input np.array of floats (x) using
    Abramowitz and Stegun method (maximum error: 1.5e−7)

    # >>> erf(0.5)  # doctest: +ELLIPSIS
    # 0.52050001...
    # >>> erf(1.0)  # doctest: +ELLIPSIS
    # 0.84270068...
    # >>> erf(-0.5)  # doctest: +ELLIPSIS
    # 0.52089577...
    """
    sign = np.ones_like(x)
    sign[x < 0] = -1.0

    x = np.abs(x)

    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
               0.254829592) * t * np.exp(-x*x)

    return sign * y
