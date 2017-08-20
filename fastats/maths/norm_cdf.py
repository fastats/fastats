
from numpy import sqrt

from fastats.maths.erfc import erfc


def norm_cdf(x, mu, sigma):
    """
    Normal Cumulative Distribution
    Function

    Norm CDF across X for a normal
    distribution with mean `mu` and
    standard deviation `sigma`.
    """
    t = x - mu
    s = sigma * sqrt(2.0)
    y = 0.5 * erfc(-t / s)
    if y > 1.0:
        y = 1.0
    return y


if __name__ == '__main__':
    import doctest
    doctest.testmod()