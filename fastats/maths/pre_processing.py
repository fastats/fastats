from numpy import empty_like, mean, std, sqrt, min, max


def standard_scale(data, ddof=0):

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

    n = data.shape[1]
    res = empty_like(data)

    for i in range(n):
        data_i = data[:, i]
        res[:, i] = (data_i - min(data_i)) / (max(data_i) - min(data_i))

    return res


if __name__ == '__main__':
    import pytest
    pytest.main()
