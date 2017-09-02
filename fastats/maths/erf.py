import numpy as np


def erf(x):
    """
    Returns error function for an input np.array of floats (x) using
    Abramowitz and Stegun method (maximum error: 1.5×10−7)
    """
    sign = np.ones_like(x)
    sign[x < 0] = -1.0

    x = np.abs(x)

    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
               0.254829592) * t * np.exp(-x*x)

    return sign * y
