
from numpy import column_stack

from fastats.utilities.pre_processing import standard_scale


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
    A = column_stack([x, y])
    return pearson_pairwise(A).diagonal(1)


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
    A_std = standard_scale(A)
    return (A_std.T @ A_std) / n


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
