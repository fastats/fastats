
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
    res = empty_like(A, dtype=float)

    for i in range(n):
        data_i = A[:, i]
        res[:, i] = (data_i - mean(data_i)) / std(data_i)

    if ddof == 0:
        return res
    elif ddof == 1:
        m = A.shape[0]
        return res * sqrt((m - 1) / m)


def min_max(A):
    """
    Standardise data by scaling data points by the sample minimum and maximum
    such that all data points lie in the range 0 to 1, equivalent to sklearn
    MinMaxScaler.
    """
    assert A.ndim > 1

    n = A.shape[1]
    res = empty_like(A, dtype=float)

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


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
