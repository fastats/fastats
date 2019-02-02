
from numba import prange
from numpy import empty_like, mean, std, sqrt, argsort, ones, nonzero, empty
from numpy import float64 as np_float64
from numpy import int32 as np_int32
from numpy import max as np_max
from numpy import min as np_min


def scale(A):
    """
    A no-op data scaling transformation
    """
    return A


def standard(A, ddof=0):
    """
    Standardise data by removing the mean and scaling to unit variance,
    equivalent to sklearn StandardScaler.

    The delta degrees of freedom (ddof) may be used to correct for bias
    in the estimation of population variance by applying Bessel's
    correction: https://en.wikipedia.org/wiki/Bessel%27s_correction
    """
    assert A.ndim > 1

    if ddof not in (0, 1):
        raise ValueError('ddof must be either 0 or 1')

    n = A.shape[1]
    res = empty_like(A, dtype=np_float64)

    for i in range(n):
        data_i = A[:, i]
        res[:, i] = (data_i - mean(data_i)) / std(data_i)

    if ddof == 1:
        m = A.shape[0]
        res *= sqrt((m - 1) / m)

    return res


def min_max(A):
    """
    Standardise data by scaling data points by the sample minimum and maximum
    such that all data points lie in the range 0 to 1, equivalent to sklearn
    MinMaxScaler.
    """
    assert A.ndim > 1

    n = A.shape[1]
    res = empty_like(A, dtype=np_float64)

    for i in range(n):
        data_i = A[:, i]
        data_min = np_min(data_i)
        res[:, i] = (data_i - data_min) / (np_max(data_i) - data_min)

    return res


def rank(A):
    """
    Rank supplied data column-wise.  Ties are dealt with using
    the average method whereby the average of the ranks that would
    have been assigned to all the tied values is assigned to each value.
    """
    assert A.ndim > 1

    A = A.astype(np_float64)  # may result in spurious ties
    res = empty_like(A)
    m, n = A.shape

    for i in range(n):
        data_i = A[:, i]
        data_i_std = empty_like(data_i, dtype=np_int32)
        sort_order = argsort(data_i)

        data_i = data_i[sort_order]
        obs = ones(m, dtype=np_int32)

        for j in range(m):
            idx = sort_order[j]
            data_i_std[idx] = j
            if j > 0:
                if data_i[j] == data_i[j - 1]:
                    obs[j] = 0

        dense = obs.cumsum()[data_i_std]

        non_zero_indices = nonzero(obs)[0]
        count = empty(len(non_zero_indices) + 1)
        count[:-1] = non_zero_indices
        count[-1] = m

        res[:, i] = (count[dense] + count[dense - 1] + 1) / 2.0

    return res


def demean(A):
    """
    Subtract the mean from the supplied data column-wise.
    """
    assert A.ndim > 1

    n = A.shape[1]
    res = empty_like(A, dtype=np_float64)

    for i in range(n):
        data_i = A[:, i]
        res[:, i] = data_i - mean(data_i)

    return res


def shrink_off_diagonals(A, shrinkage_factor):
    """
    Given a square matrix A, apply a multiplication
    factor to all off-diagonal elements - e.g. to
    shrink off-diagonal correlation / covariance.

    Example usage:

    >>> import numpy as np
    >>> A = np.array([[1.1, 0.9, 0.8], [0.9, 1.2, 0.9], [0.8, 0.9, 1.3]])
    >>> shrink_off_diagonals(A, 0.1)
    array([[ 1.1 ,  0.09,  0.08],
           [ 0.09,  1.2 ,  0.09],
           [ 0.08,  0.09,  1.3 ]])
    """
    m, n = A.shape
    assert m == n

    out = empty_like(A)

    for i in range(n):
        for j in range(n):
            val = A[i, j]
            if i != j:
                val *= shrinkage_factor
            out[i, j] = val

    return out

# ------------------------------------------------------------------------------------------
# explicitly parallel versions - numba.prange indicates explicit parallel loop opportunities
# ------------------------------------------------------------------------------------------


def standard_parallel(A, ddof=0):
    """
    Standardise data by removing the mean and scaling to unit variance,
    equivalent to sklearn StandardScaler.

    The delta degrees of freedom (ddof) may be used to correct for bias
    in the estimation of population variance by applying Bessel's
    correction: https://en.wikipedia.org/wiki/Bessel%27s_correction

    Uses explicit parallel loop; may offer improved performance in some
    cases.
    """
    assert A.ndim > 1
    assert ddof == 0 or ddof == 1

    n = A.shape[1]
    res = empty_like(A, dtype=np_float64)

    for i in prange(n):
        data_i = A[:, i]
        res[:, i] = (data_i - mean(data_i)) / std(data_i)

    if ddof == 1:
        m = A.shape[0]
        res *= sqrt((m - 1) / m)

    return res


def min_max_parallel(A):
    """
    Standardise data by scaling data points by the sample minimum and maximum
    such that all data points lie in the range 0 to 1, equivalent to sklearn
    MinMaxScaler.

    Uses explicit parallel loop; may offer improved performance in some
    cases.
    """
    assert A.ndim > 1

    n = A.shape[1]
    res = empty_like(A, dtype=np_float64)

    for i in prange(n):
        data_i = A[:, i]
        data_min = np_min(data_i)
        res[:, i] = (data_i - data_min) / (np_max(data_i) - data_min)

    return res


def demean_parallel(A):
    """
    Subtract the mean from the supplied data column-wise.

    Uses explicit parallel loop; may offer improved performance in some
    cases.
    """
    assert A.ndim > 1

    n = A.shape[1]
    res = empty_like(A, dtype=np_float64)

    for i in prange(n):
        data_i = A[:, i]
        res[:, i] = data_i - mean(data_i)

    return res


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
