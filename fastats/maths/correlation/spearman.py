
import numpy as np

from fastats.maths.correlation.pearson import pearson_pairwise
from fastats.scaling.scaling import rank


def spearman(x, y):
    """
    Calculates the Spearman rank correlation
    coefficient for the inputs `x` and `y`.

    If there are any Nan values in the data,
    they will be ignored in the rank for that
    variable, and therefore may skew the results.
    See the test_spearman unit-tests for an
    example.

    The provided inputs must have no ties.

    Example
    -------

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([2, 3, 4, 3])
    >>> spearman(x, y)
    0.8

    :param x: A `numpy.array` of floats or ints
    :param y: A `numpy.array` of floats or ints
    :return: A float representing the correlation
    """
    assert x.shape == y.shape

    n = len(x)
    rank_x = np.argsort(x)
    rank_y = np.argsort(y)

    d = rank_x - rank_y
    d2 = d ** 2
    return 1 - (6 * np.sum(d2)) / (n**3 - n)


def spearman_pairwise(A):
    """
    Calculates the Spearman rank correlation
    coefficient between pairs of columns of
    the supplied matrix A (similar to
    pandas.DataFrame.corr(method='spearman').

    Ties are dealt with using the 'average'
    method, as described in rank_data.
    """
    assert A.ndim > 1
    assert A.shape[1] > 1

    A_rank = rank(A)
    return pearson_pairwise(A_rank)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
