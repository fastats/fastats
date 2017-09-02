
import numpy as np
from numpy import sum, power, abs, mean


def sum_sq_dev(x):
    """
    Sum of squared deviations.

    :param x: Numpy array
    :return: Scalar of type `x.dtype`

    >>> x = np.array([205.,195.,210.,340.,299.,
    ...               230.,270.,243.,340.,240.])
    >>> sum_sq_dev(x) # doctest: +ELLIPSIS
    25681.60000...
    """
    return sum(power(abs(x - mean(x)), 2))


if __name__ == '__main__':
    import pytest
    pytest.main()
