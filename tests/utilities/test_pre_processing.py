from unittest import TestCase

import numpy as np
import pandas as pd
from pytest import raises, mark
from scipy.stats import rankdata
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fastats.utilities.pre_processing import standard_scale, min_max_scale, rank_data
from tests.data.datasets import SKLearnDataSets


class SKLearnTestMixin:

    def test_scaled_values_versus_sklearn(self):
        predictors = load_iris().data
        scaler = self._scaler()

        expected = scaler.fit_transform(predictors)
        output = self._func(predictors)
        assert np.allclose(expected, output)


class StandardScaleTests(TestCase, SKLearnTestMixin):

    def setUp(self):
        super().setUp()
        self._func = standard_scale
        self._scaler = StandardScaler

    def check_zscore(self, ddof):
        predictors = load_iris().data
        df = pd.DataFrame(predictors)

        def zscore(data):
            return (data - data.mean()) / data.std(ddof=ddof)

        expected = df.apply(zscore)
        output = self._func(predictors, ddof=ddof)
        assert np.allclose(expected, output)

    def test_standard_scale_ddof_zero(self):
        self.check_zscore(ddof=0)

    def test_standard_scale_ddof_one(self):
        self.check_zscore(ddof=1)

    def test_standard_scale_raise_if_ddof_not_one_or_zero(self):
        predictors = load_iris().data

        for ddof in -1, 2:
            with raises(ValueError, message='ddof must be either 0 or 1'):
                _ = self._func(predictors, ddof=ddof)


class MinMaxScaleTests(TestCase, SKLearnTestMixin):

    def setUp(self):
        super().setUp()
        self._func = min_max_scale
        self._scaler = MinMaxScaler


@mark.parametrize('A', SKLearnDataSets)
def test_rank_data(A):
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
