
import numpy as np
import pandas as pd
import pytest

from fastats.maths.ewma import ewma, ewma_2d


def _validate_results(random_data, fn=ewma):
    df = pd.DataFrame(random_data)
    pandas_result = df.ewm(halflife=10).mean()
    ewma_result = fn(random_data, halflife=10)
    fast_result = pd.DataFrame(ewma_result)
    pd.testing.assert_frame_equal(pandas_result, fast_result)


def test_ewma_1d_array():
    rng = np.random.RandomState(0)
    random_data = rng.randn(100)
    _validate_results(random_data)


@pytest.mark.parametrize('fn', (ewma, ewma_2d))
def test_ewma_basic_sanity(fn):
    rng = np.random.RandomState(0)
    random_data = rng.randn(10000).reshape(1000, 10)
    _validate_results(random_data, fn)


@pytest.mark.parametrize('fn', (ewma, ewma_2d))
def test_bad_halflifes(fn):
    random_data = np.random.random(100)
    bad_halflifes = [
        np.NaN,
        np.inf,
        -np.inf,
        -100,
        0,
    ]
    for halflife in bad_halflifes:
        with pytest.raises(AssertionError):
            fn(random_data, halflife)


@pytest.mark.xfail(reason='NaN support to be implemented')
def test_nan_compat():
    random_data = np.random.random((100, 100))
    random_data[0, :] = np.nan
    random_data[10, :] = np.nan
    random_data[70, :50] = np.nan
    _validate_results(random_data)


if __name__ == '__main__':
    pytest.main([__file__])
