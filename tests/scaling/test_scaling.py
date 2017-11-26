import numpy as np
import pandas as pd
from pytest import mark
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fastats.scaling.scaling import standard_scale, min_max_scale, rank_data
from tests.data.datasets import SKLearnDataSets


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_versus_sklearn(A):
    data = A.value.data
    expected = StandardScaler().fit_transform(data)
    output = standard_scale(data)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_min_max_scale_versus_sklearn(A):
    data = A.value.data
    expected = MinMaxScaler().fit_transform(data)
    output = min_max_scale(data)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_with_bessel_correction(A):
    data = A.value.data
    df = pd.DataFrame(data)

    def zscore(data):
        return (data - data.mean()) / data.std(ddof=1)

    expected = df.apply(zscore).values

    output = standard_scale(data, ddof=1)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_rank_data_versus_scipy(A):
    data = A.value.data

    # rank the data all at once
    output = rank_data(data)

    # check each column versus scipy equivalent
    for i in range(data.shape[1]):
        feature = data[:, i]
        expected = rankdata(feature)
        assert np.allclose(expected, output[:, i])


if __name__ == '__main__':
    import pytest
    pytest.main()
