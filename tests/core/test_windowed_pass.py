
import numpy as np
from pytest import approx

from fastats import windowed_pass, windowed_pass_2d
from fastats.linear_algebra import ols, r_squared


def std(x):
    return np.std(x)


def test_windowed_pass_basic_sanity():
    assert std([1, 2, 3]) == approx(0.816496580)

    data = np.array(range(100), dtype='float') ** 2

    res = windowed_pass(data, 5, value=std)

    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isnan(res[2])
    assert np.isnan(res[3])
    assert res[4] == approx(5.89915248)
    assert res[5] == approx(8.64869932)
    assert res[-1] == approx(274.36253389)

    raw = windowed_pass(data, 5)
    assert np.isnan(raw[3])
    assert raw[4] == approx(0.0)
    assert raw[50] == approx(2116.0)


def mean(x):
    return np.sum(x) / x.size


def test_windowed_pass_mean_size_5():
    assert mean(np.array([1, 2, 3])) == approx(2.0)

    data = np.array(range(100), dtype='float') ** 2

    res = windowed_pass(data, 5, value=mean)

    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert np.isnan(res[3])
    assert res[4] == approx(6.0)
    assert res[5] == approx(11.0)
    assert res[8] == approx(38.0)
    assert res[-1] == approx(9411.0)


def test_windowed_pass_mean_size_10():
    data = np.array(range(100), dtype='float') ** 2

    res = windowed_pass(data, 10, value=mean)

    assert np.all(np.isnan(res[:9]))
    assert res[9] == approx(28.5)
    assert res[10] == approx(38.5)
    assert res[11] == approx(50.5)
    assert res[-1] == approx(8938.5)


def nsum(x):
    return np.sum(x)


def test_windowed_pass_sum():
    data = np.array(range(100), dtype='float')
    assert nsum(data) == approx(4950)

    res = windowed_pass(data, 3, value=nsum)

    assert np.isnan(res[0])
    assert np.isnan(res[1])
    assert res[2] == approx(3.0)
    assert res[3] == approx(6.0)
    assert res[4] == approx(9.0)
    assert res[-1] == approx(294.0)

    res5 = windowed_pass(data, 5, value=nsum)

    assert np.all(np.isnan(res5[:4]))
    assert res5[5] == approx(15.0)
    assert res5[-1] == approx(485.0)


def nanmean(x):
    return np.nanmean(x)


def test_windowed_pass_nanmean():
    raw = [1, 2, 3, np.nan, np.nan, 4, 5, 6, 7, np.nan]
    data = np.array(raw, dtype='float')
    assert nanmean(data) == approx(4.0)

    res = windowed_pass(data, 5, value=nanmean)

    assert np.all(np.isnan(res[:4]))
    assert res[4] == approx(2.0)
    assert res[5] == approx(3.0)
    assert res[6] == approx(4.0)
    assert res[7] == approx(5.0)
    assert res[8] == approx(5.5)

    res2 = windowed_pass(data, 2, value=nanmean)

    assert np.isnan(res2[0])
    assert res2[1] == approx(1.5)
    assert res2[2] == approx(2.5)
    assert res2[3] == approx(3.0)
    assert res2[-1] == approx(7.0)


def nanmedian(x):
    return np.nanmedian(x)


def test_windowed_pass_nanmedian():
    raw = [1, 2, 3, np.nan, np.nan, 4, 5, 6, 7, np.nan]
    data = np.array(raw, dtype='float')
    assert nanmedian(data) == approx(4.0)

    res = windowed_pass(data, 5, value=nanmedian)

    assert np.all(np.isnan(res[:4]))
    assert res[4] == approx(2.0)
    assert res[5] == approx(3.0)
    assert res[6] == approx(4.0)
    assert res[7] == approx(5.0)
    assert res[8] == approx(5.5)

    res2 = windowed_pass(data, 2, value=nanmedian)
    assert np.isnan(res2[0])
    assert res2[1] == approx(1.5)
    assert res2[2] == approx(2.5)
    assert res2[3] == approx(3.0)
    assert np.isnan(res2[4])
    assert res2[5] == approx(4.0)
    assert res2[6] == approx(4.5)
    assert res2[7] == approx(5.5)
    assert res2[8] == approx(6.5)
    assert res2[9] == approx(7.0)


def ols_wrap(x):
    a = x[:, :1]
    b = x[:, 1:]
    return ols(a, b)[0][0]


def test_windowed_pass_ols_wrapped():
    """
    Returns a 2-column array the same size
    as `data` with the OLS slope values
    duplicated in both columns.
    """
    data = np.array([
        [0, 1],  # np.nan
        [1, 2],  # np.nan
        [2, 3],  # 1.40000000
        [3, 4],  # 1.42857143
        [2, 3],  # 1.41176471
        [1, 2],  # 1.42857143
        [0, 1]   # 1.60000000
    ], dtype='float')
    assert ols_wrap(data) == approx(1.4736842105263157)

    res = windowed_pass(data, 3, value=ols_wrap)

    # Two values are returned here as we haven't
    # added support for multiple args yet
    assert np.isnan(res[0, 0])
    assert np.isnan(res[1, 0])
    assert res[2, 0] == approx(1.6)
    assert res[3, 0] == approx(1.42857143)
    assert res[4, 0] == approx(1.41176471)
    assert res[5, 0] == approx(1.42857143)
    assert res[6, 0] == approx(1.6)


def test_windowed_pass_2d_basic_sanity():
    data = np.array([[0.0, 1.0], [1.0, 2.0], [3.0, 4.0]])
    result = windowed_pass_2d(data, 2)

    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])
    assert result[1, 0] == approx(0.0)
    assert result[1, 1] == approx(1.0)
    assert result[2, 0] == approx(1.0)
    assert result[2, 1] == approx(2.0)


def ols_r_squared(x):
    out = np.zeros(2)
    a = x[:, :1]
    b = x[:, 1:]
    slope = ols(a, b)[0][0]
    r2 = r_squared(a, b)[0][0]
    out[0] = slope
    out[1] = r2
    return out


def test_windowed_pass_ols_r_squared():
    """
    Returns a 2-column array the same size
    as `data` with the OLS slope values
    in the first column and the r^2 values
    in the second.
    """
    data = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [2, 3],
        [1, 2],
        [0, 1]
    ], dtype='float')

    raw = ols_r_squared(data)
    assert raw[0] == approx(1.47368421)
    assert raw[1] == approx(0.63157895)

    res = windowed_pass(data, 3, value=ols_r_squared)

    assert np.isnan(res[0, 0])
    assert np.isnan(res[0, 1])
    assert np.isnan(res[1, 0])
    assert np.isnan(res[1, 1])
    assert res[2, 0] == approx(1.6)
    assert res[2, 1] == approx(0.4)
    assert res[3, 0] == approx(1.42857143)
    assert res[3, 1] == approx(0.78571429)
    assert res[4, 0] == approx(1.41176471)
    assert res[4, 1] == approx(0.82352941)
    assert res[5, 0] == approx(1.42857143)
    assert res[5, 1] == approx(0.78571429)
    assert res[6, 0] == approx(1.6)
    assert res[6, 1] == approx(0.4)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
