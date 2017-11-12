from unittest import TestCase

import numpy as np
import pandas as pd
from pytest import raises
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from fastats.maths.pre_processing import standard_scale, min_max_scale


class RollingStatsTests:

    def test_scaled_values(self):
        predictors = load_iris().data
        scaler = self._scaler()

        expected = scaler.fit_transform(predictors)
        output = self._func(predictors)
        assert np.allclose(expected, output)


class StandardScaleTests(TestCase, RollingStatsTests):

    def setUp(self):
        super().setUp()
        self._func = standard_scale
        self._scaler = StandardScaler

    def test_standard_scale_ddof_one(self):
        predictors = load_iris().data
        df = pd.DataFrame(predictors)

        def zscore(data):
            return (data - data.mean()) / data.std(ddof=1)  # pandas default

        expected = df.apply(zscore)
        output = self._func(predictors, ddof=1)
        assert np.allclose(expected, output)

    def test_standard_scale_raise_if_ddof_not_one_or_zero(self):
        predictors = load_iris().data

        for ddof in -1, 2:
            with raises(ValueError, message='ddof must be either 0 or 1'):
                _ = self._func(predictors, ddof=ddof)


class MinMaxScaleTests(TestCase, RollingStatsTests):

    def setUp(self):
        super().setUp()
        self._func = min_max_scale
        self._scaler = MinMaxScaler


if __name__ == '__main__':
    import pytest
    pytest.main()
