
import numpy as np
import pandas as pd
from numba import njit
from pytest import mark, raises
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fastats.scaling.scaling import (standard, min_max, rank, scale, demean, standard_parallel, min_max_parallel,
                                     demean_parallel)
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
def test_standard_scale_with_bessel_correction_versus_sklearn(A):
    data = A.value.data
    df = pd.DataFrame(data)

    def zscore(data):
        return (data - data.mean()) / data.std(ddof=1)

    expected = df.apply(zscore).values

    output = standard(data, ddof=1)
    assert np.allclose(expected, output)


def test_standard_scale_raises_if_ddof_ne_0_or_1():
    data = np.arange(20, dtype=float).reshape(2, 10)

    for ddof in -1, 2:
        with raises(ValueError):
            _ = standard(data, ddof=ddof)


@mark.parametrize('A', SKLearnDataSets)
def test_min_max_scale_versus_sklearn(A):
    data = A.value.data
    expected = MinMaxScaler().fit_transform(data)
    output = min_max(data)
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


@mark.parametrize('A', SKLearnDataSets)
def test_demean(A):
    data = A.value.data
    expected = data - data.mean(axis=0)
    output = demean(data)
    assert np.allclose(expected, output)


# ---------------------------------
# explicitly parallel version tests
# ---------------------------------
demean_parallel_jit = njit(demean_parallel, parallel=True)
min_max_parallel_jit = njit(min_max_parallel, parallel=True)
standard_parallel_jit = njit(standard_parallel, parallel=True)


@mark.parametrize('A', SKLearnDataSets)
def test_demean_parallel(A):
    data = A.value.data
    expected = data - data.mean(axis=0)

    for fn in demean_parallel, demean_parallel_jit:
        output = fn(data)
        assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_min_max_scale_parallel_versus_sklearn(A):
    data = A.value.data
    expected = MinMaxScaler().fit_transform(data)

    for fn in min_max_parallel, min_max_parallel_jit:
        output = fn(data)
        assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_parallel_versus_sklearn(A):
    data = A.value.data
    expected = StandardScaler().fit_transform(data)

    for fn in standard_parallel, standard_parallel_jit:
        output = fn(data)
        assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_parallel_with_bessel_correction_versus_sklearn(A):
    data = A.value.data
    df = pd.DataFrame(data)

    def zscore(data):
        return (data - data.mean()) / data.std(ddof=1)

    expected = df.apply(zscore).values

    for fn in standard_parallel, standard_parallel_jit:
        output = fn(data, ddof=1)
        assert np.allclose(expected, output)


def test_standard_scale_parallel_raises_if_ddof_ne_0_or_1():
    data = np.arange(20, dtype=float).reshape(2, 10)

    for fn in standard_parallel, standard_parallel_jit:
        with raises(ValueError):
            _ = fn(data, ddof=-1)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
