
import numpy as np

from fastats.scaling import standard


def pearson(x, y):
    """
    Calculates the pearson r correlation to
    measure the degree of the relationship
    between two linearly related variables.

    If there are any NaN values in the data,
    the result will be NaN.

    Example
    -------

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([2, 3, 4, 3])
    >>> pearson(x, y)
    0.63245553203367588

    :param x: a `numpy.array` of floats or ints
    :param y: a `numpy.array` of floats or ints
    :return: a float representing the correlation
    """
    assert x.shape == y.shape

    n = len(x)
    xy = x * y
    x2 = x ** 2
    y2 = y ** 2

    sum_x = np.sum(x)
    sum_y = np.sum(y)

    numer = n * np.sum(xy) - sum_x * sum_y

    first = n * np.sum(x2) - sum_x ** 2
    second = n * np.sum(y2) - sum_y ** 2
    denom = np.sqrt(first * second)
    return numer / denom


def pearson_pairwise(A):
    """
    Calculates the pearson r correlation to
    measure the degree of the relationship
    between pairs of columns of the supplied
    matrix A (similar to pandas.DataFrame.corr)
    """
    assert A.ndim > 1
    assert A.shape[1] > 1

    n = A.shape[0]
    A_std = standard(A)
    return (A_std.T @ A_std) / n


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
