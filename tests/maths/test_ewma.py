
import numpy as np
import pandas as pd
import pytest

from fastats.maths.ewma import ewma


def _validate_results(random_data):
    df = pd.DataFrame(random_data)
    pandas_result = df.ewm(halflife=10).mean()
    ewma_result = ewma(random_data, halflife=10)
    fast_result = pd.DataFrame(ewma_result)
    pd.testing.assert_frame_equal(pandas_result, fast_result)


def test_ewma_1d_array():
    random_data = np.random.random(100)
    _validate_results(random_data)


def test_ewma_basic_sanity():
    random_data = np.random.random((100, 100))
    _validate_results(random_data)


def test_bad_halflifes():
    random_data = np.random.random(100)
    bad_halflifes = [
        np.NaN,
        np.inf,
        -np.inf,
        -100
    ]
    for halflife in bad_halflifes:
        with pytest.raises(AssertionError):
            ewma(random_data, halflife)


@pytest.mark.xfail(reason='NaN support to be implemented')
def test_nan_compat():
    random_data = np.random.random((100, 100))
    random_data[0, :] = np.nan
    random_data[10, :] = np.nan
    random_data[70, :50] = np.nan
    _validate_results(random_data)


if __name__ == '__main__':
    pytest.main([__file__])
