
from numpy import abs, sqrt, exp, pi


def norm_pdf(x, mu, sigma):
    """
    Normal distribution Probability
    Density Function.

    PDF across `x` for a normal
    distribution with mean `mu` and
    standard deviation `sigma`.
    """
    a = abs(sigma)
    u = (x - mu) / a
    v = sqrt(2 * pi) * a
    y = (1 / v) * exp(-u * u / 2)
    return y


if __name__ == '__main__':
    import doctest
    doctest.testmod()
