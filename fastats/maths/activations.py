
from numpy import exp, log, maximum


def relu(x):
    """
    Implements the ReLU (rectifier) function

    :param x: A number
    :return: A number of the same type

    Tests
    -----
    >>> relu(-0.5)
    0.0
    >>> relu(1.0)
    1.0
    >>> relu(0)    # maintains input type
    0
    >>> relu(0.0)
    0.0
    >>> relu(np.nan)   # NaN should work just fine
    nan
    """
    return maximum(x, 0)


def softplus(x):
    """
    Implements the softplus function

    :param x: A number
    :return: A float

    Tests
    -----
    >>> softplus(-100)   # doctest: +ELLIPSIS
    0.0...
    >>> softplus(100)   # doctest: +ELLIPSIS
    100.0...
    >>> softplus(0)  # doctest: +ELLIPSIS
    0.6931471...
    >>> softplus(np.nan)   # NaN should work just fine
    nan
    """
    return log(1 + exp(x))


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
