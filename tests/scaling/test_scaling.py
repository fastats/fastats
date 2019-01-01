
import numpy as np
import pandas as pd
import sys
from numba import njit
from numpy.testing import assert_allclose
from pytest import mark, raises, approx
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fastats.scaling.scaling import (standard, min_max, rank, scale, demean, standard_parallel, min_max_parallel,
                                     demean_parallel, shrink_off_diagonals)
from tests.data.datasets import SKLearnDataSets


def test_scale_no_op():
    data = np.arange(100, dtype=float)
    data[13] = np.nan
    output = scale(data)
    assert np.allclose(data, output, equal_nan=True)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_versus_sklearn(A):
    data = A.value
    expected = StandardScaler().fit_transform(data)
    output = standard(data)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_with_bessel_correction_versus_sklearn(A):
    data = A.value
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
    data = A.value
    expected = MinMaxScaler().fit_transform(data)
    output = min_max(data)
    assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_rank_scale_versus_scipy(A):
    data = A.value

    # rank the data all at once
    output = rank(data)

    # check each column versus scipy equivalent
    for i in range(data.shape[1]):
        feature = data[:, i]
        expected = rankdata(feature)
        assert np.allclose(expected, output[:, i])


@mark.parametrize('A', SKLearnDataSets)
def test_demean(A):
    data = A.value
    expected = data - data.mean(axis=0)
    output = demean(data)
    assert np.allclose(expected, output)


@mark.parametrize('factor', np.linspace(-1, 1, 9), ids='factor_{0:.2f}'.format)
def test_shrink_off_diagonals(factor):

    A = np.empty(shape=(10, 10))
    m, n = A.shape

    for i in range(m):
        for j in range(n):
            A[i, j] = 1.0 - abs(i - j) / m

    output = shrink_off_diagonals(A, factor)

    # diagonals should be unaffected
    assert_allclose(np.diag(output), np.diag(A))

    # all other values should have been shrunk
    for i in range(m):
        for j in range(n):
            if i != j:
                assert output[i, j] == approx(A[i, j] * factor)


def test_shrink_off_diagonals_factor_zero():
    A = np.arange(100, dtype=np.float64).reshape(10, 10)

    # special case where factor is 0 - we expect an output
    # where all off diagonal values are zeroed out
    output = shrink_off_diagonals(A, 0)
    assert_allclose(output, np.diag(np.diag(A)))


# ----------------------------------------------------------------
# explicitly parallel algorithm tests
#
# Note: parallel not supported on 32bit platforms
# ----------------------------------------------------------------

parallel = not (sys.platform == 'win32')

demean_parallel_jit = njit(demean_parallel, parallel=parallel)
min_max_parallel_jit = njit(min_max_parallel, parallel=parallel)
standard_parallel_jit = njit(standard_parallel, parallel=parallel)


@mark.parametrize('A', SKLearnDataSets)
def test_demean_parallel(A):
    data = A.value
    expected = data - data.mean(axis=0)

    for fn in demean_parallel, demean_parallel_jit:
        output = fn(data)
        assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_min_max_scale_parallel_versus_sklearn(A):
    data = A.value
    expected = MinMaxScaler().fit_transform(data)

    for fn in min_max_parallel, min_max_parallel_jit:
        output = fn(data)
        assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_parallel_versus_sklearn(A):
    data = A.value
    expected = StandardScaler().fit_transform(data)

    for fn in standard_parallel, standard_parallel_jit:
        output = fn(data)
        assert np.allclose(expected, output)


@mark.parametrize('A', SKLearnDataSets)
def test_standard_scale_parallel_with_bessel_correction_versus_sklearn(A):
    data = A.value
    df = pd.DataFrame(data)

    def zscore(data):
        return (data - data.mean()) / data.std(ddof=1)

    expected = df.apply(zscore).values

    # Issues seen here running standard_parallel_jit using
    # numba 0.35 on OS X.
    # The standard parallel variant works fine, but the
    # jit version is returning garbage float values for
    # some (not all) data sets.
    # Looks very much like a threading issue.
    for fn in (standard_parallel, standard_parallel_jit):
        output = fn(data, ddof=1)
        assert np.allclose(expected, output)


def test_standard_scale_parallel_raises_if_ddof_ne_0_or_1():
    data = np.arange(20, dtype=float).reshape(2, 10)

    for fn in standard_parallel, standard_parallel_jit:
        with raises(AssertionError):
            _ = fn(data, ddof=-1)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
