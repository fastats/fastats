
import numpy as np
from numpy import power, mean


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
    population_mean = mean(x)
    total = 0.0

    for value in x:
        total += power((value - population_mean), 2)

    return total


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
