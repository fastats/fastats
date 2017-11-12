
import numpy as np
from pytest import approx

from fastats import windowed_pass


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


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
