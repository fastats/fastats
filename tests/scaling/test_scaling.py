import numpy as np
import pandas as pd
from pytest import mark
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fastats.scaling.scaling import standard, min_max, rank, scale
from tests.data.datasets import SKLearnDataSets


def test_scale_no_op():
    data = np.arange(100, dtype=float)
    data[13] = np.nan
    output = scale(data)
    assert np.allclose(data, output, equal_nan=True)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_versus_sklearn(A):
    data = A.value.data
    expected = StandardScaler().fit_transform(data)
    output = standard(data)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_min_max_scale_versus_sklearn(A):
    data = A.value.data
    expected = MinMaxScaler().fit_transform(data)
    output = min_max(data)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_with_bessel_correction(A):
    data = A.value.data
    df = pd.DataFrame(data)

    def zscore(data):
        return (data - data.mean()) / data.std(ddof=1)

    expected = df.apply(zscore).values

    output = standard(data, ddof=1)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_rank_scale_versus_scipy(A):
    data = A.value.data

    # rank the data all at once
    output = rank(data)

    # check each column versus scipy equivalent
    for i in range(data.shape[1]):
        feature = data[:, i]
        expected = rankdata(feature)
        assert np.allclose(expected, output[:, i])


if __name__ == '__main__':
    import pytest
    pytest.main()
