
from numpy import exp


def logistic(x):
    """
    Implements the logistic function.

    Could also be implemented as:
    (1 + tanh(x/2)) / 2
    May be worth seeing if there's
    performance/numerical differences.

    :param x: A float
    :return: A float

    Tests
    -----
    >>> logistic(0.0)
    0.5
    >>> logistic(1.0) # doctest: +ELLIPSIS
    0.73105857...
    >>> logistic(-1.0) # doctest: +ELLIPSIS
    0.2689414...
    >>> logistic(-2.0) # doctest: +ELLIPSIS
    0.1192029...
    """
    return 1 / (1 + exp(-x))


if __name__ == '__main__':
    import pytest
    pytest.main()