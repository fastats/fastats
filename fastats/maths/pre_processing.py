from numpy import empty_like, mean, std, sqrt, min, max


def standard_scale(data, ddof=0):
    """
    Standardise data by removing the mean and scaling to unit variance.

    The delta degrees of freedom (ddof) may be used to correct for bias
    in the estimation of population variance; the default is 0 per sklearn
    StandardScaler but may be set to 1 to apply Bessel's correction.

    https://en.wikipedia.org/wiki/Bessel%27s_correction
    """
    if ddof not in (0, 1):
        raise ValueError('ddof must be either 0 or 1')

    n = data.shape[1]
    res = empty_like(data, dtype=float)

    for i in range(n):
        data_i = data[:, i]
        res[:, i] = (data_i - mean(data_i)) / std(data_i)

    if ddof == 0:
        return res
    elif ddof == 1:
        m = data.shape[0]
        return res * sqrt((m - 1) / m)


def min_max_scale(data):
    """
    Standardise data by scaling data points such that they lie in the range
    0 to 1 by subtracting the minimum value and diving through by the
    difference in maximum and minimum values.
    """
    n = data.shape[1]
    res = empty_like(data, dtype=float)

    for i in range(n):
        data_i = data[:, i]
        res[:, i] = (data_i - min(data_i)) / (max(data_i) - min(data_i))

    return res


if __name__ == '__main__':
    import pytest
    pytest.main()
