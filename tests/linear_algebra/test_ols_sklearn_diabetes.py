
from unittest import TestCase

import numpy as np
import statsmodels.api as sm
from pytest import approx, mark
from sklearn import datasets

from fastats.linear_algebra import (
    ols, ols_cholesky, ols_qr, ols_svd,
    add_intercept, adjusted_r_squared,
    adjusted_r_squared_no_intercept, fitted_values,
    mean_standard_error_residuals, r_squared,
    r_squared_no_intercept, residuals, standard_error,
    sum_of_squared_residuals, t_statistic, f_statistic,
    f_statistic_no_intercept, drop_missing
)

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit


drop_missing_jit = convert_to_jit(drop_missing)


class BaseOLS(TestCase):
    def setUp(self):
        self._data = datasets.load_diabetes()
        self._labels =[
            'age', 'sex', 'bmi', 'bp', 's1',
            's2', 's3', 's4', 's5', 's6'
        ]


class SklearnDiabetesOLS:
    """
    Linear Regression example taken from the
    fast.ai course 'Numerical Linear Algebra'
    """
    def test_coefficients(self):
        data, target = self._data.data, self._data.target
        coeffs = self._func(data, target)

        expected = np.array([
            -10.01219782, -239.81908937, 519.83978679, 324.39042769,
            -792.18416163,  476.74583782,  101.04457032,  177.06417623,
            751.27932109,   67.62538639
        ])

        assert np.allclose(expected, coeffs)


class OLSNaiveTests(BaseOLS, SklearnDiabetesOLS):
    def setUp(self):
        super().setUp()
        self._func = ols


class OLSQRTests(BaseOLS, SklearnDiabetesOLS):
    def setUp(self):
        super().setUp()
        self._func = ols_qr


class OLSCholeskyTests(BaseOLS, SklearnDiabetesOLS):
    def setUp(self):
        super().setUp()
        self._func = ols_cholesky


class OLSSVDTests(BaseOLS, SklearnDiabetesOLS):
    def setUp(self):
        super().setUp()
        self._func = ols_svd


class OLSFitMeasuresTestMixin:

    @staticmethod
    def fit_statsmodels_ols(A, b):
        return sm.OLS(b, A).fit()

    def get_fixtures(self):
        return self._predictors, self._targets, self._model

    def test_sum_of_squared_residuals(self):
        A, b, model = self.get_fixtures()
        expected = model.ssr
        output = sum_of_squared_residuals(A, b)
        assert output == approx(expected)

    def test_fitted_values(self):
        A, b, model = self.get_fixtures()
        expected = model.fittedvalues
        output = fitted_values(A, b)
        assert np.allclose(output, expected)

    def test_residuals(self):
        A, b, model = self.get_fixtures()
        expected = model.resid
        output = residuals(A, b)
        assert np.allclose(output, expected)

    def test_standard_error(self):
        A, b, model = self.get_fixtures()
        expected = model.bse
        output = standard_error(A, b)
        assert np.allclose(output, expected)

    def test_mean_standard_error_residuals(self):
        A, b, model = self.get_fixtures()
        output = mean_standard_error_residuals(A, b)
        expected = model.mse_resid
        assert np.allclose(output, expected)

    def test_t_statistic(self):
        A, b, model = self.get_fixtures()
        expected = model.tvalues
        output = t_statistic(A, b)
        assert np.allclose(output, expected)


class OLSModelWithoutIntercept(BaseOLS, OLSFitMeasuresTestMixin):

    def setUp(self):
        super().setUp()
        self._targets = self._data.target
        self._predictors = self._data.data
        self._model = self.fit_statsmodels_ols(self._predictors, self._targets)

    def test_r_squared(self):
        A, b, model = self.get_fixtures()
        expected = model.rsquared
        output = r_squared_no_intercept(A, b)  # this is a replica of statsmodels / R behaviour
        assert output == approx(expected)

    def test_adjusted_r_squared(self):
        A, b, model = self.get_fixtures()
        expected = model.rsquared_adj
        output = adjusted_r_squared_no_intercept(A, b)
        assert output == approx(expected)

    def test_f_statistic(self):
        A, b, model = self.get_fixtures()
        expected = model.fvalue
        output = f_statistic_no_intercept(A, b)  # this is a replica of statsmodels behaviour
        assert np.allclose(output, expected)


class OLSModelWithIntercept(BaseOLS, OLSFitMeasuresTestMixin):

    def setUp(self):
        super().setUp()
        self._targets = self._data.target
        self._predictors = sm.add_constant(self._data.data)
        self._model = self.fit_statsmodels_ols(self._predictors, self._targets)

    def test_add_intercept(self):
        A = self._data.data
        output = add_intercept(A)
        expected = sm.add_constant(A)
        assert np.allclose(output, expected)

    def test_r_squared(self):
        A, b, model = self.get_fixtures()
        expected = model.rsquared
        output = r_squared(A, b)
        assert output == approx(expected)

    def test_adjusted_r_squared(self):
        A, b, model = self.get_fixtures()
        expected = model.rsquared_adj
        output = adjusted_r_squared(A, b)
        assert output == approx(expected)

    def test_f_statistic(self):
        A, b, model = self.get_fixtures()
        expected = model.fvalue
        output = f_statistic(A, b)
        assert np.allclose(output, expected)


def test_drop_missing():

    A = np.array([[1.1, 1.2, 1.3],
                  [1.2, 1.0, 1.3],
                  [1.6, np.nan, 2.0],  # <- expect to be dropped
                  [4.5, 4.2, 4.3],
                  [4.4, 4.0, 4.2]])

    b = np.array([1.0,
                  6.0,
                  2.0,
                  3.0,
                  np.nan])  # <- expect to be dropped

    for fn in drop_missing, drop_missing_jit:
        A_bar, b_bar = fn(A, b)

        expected_A_bar = np.array([[1.1, 1.2, 1.3],
                                   [1.2, 1.0, 1.3],
                                   [4.5, 4.2, 4.3]])

        expected_b_bar = np.array([1.0,
                                   6.0,
                                   3.0])

        assert np.allclose(A_bar, expected_A_bar)
        assert np.allclose(b_bar, expected_b_bar)


def test_ols_drop_missing_versus_statsmodels():

    dataset = datasets.load_iris()
    A = dataset.data
    b = dataset.target.astype(np.float64)  # cast as float as we will set some values to NaN

    # insert some NaNs into the features
    A[1, 2] = np.nan
    A[20, 3] = np.nan

    # insert some NaNs into the targets
    b[13] = np.nan
    b[140] = np.nan

    sm_model = sm.OLS(b, A, missing='drop').fit()

    for fn in drop_missing, drop_missing_jit:
        output = ols(*fn(A, b))
        assert np.allclose(output, sm_model.params)

        output = fitted_values(*fn(A, b))
        assert np.allclose(output, sm_model.fittedvalues)


@mark.xfail(reason='Perfect multicollinearity')
def test_ols_fails_as_features_perfect_multicollinear():

    A = np.array([[1, 1, 2],
                  [1, 2, 4],
                  [2, 3, 6],
                  [3, 4, 8]])
    #                    \____ this feature is 2 * previous feature

    b = np.array([0, 1, 2, 2])

    output = ols(A, b)
    assert output is not None


@mark.xfail(reason='Feature all zero')
def test_ols_fails_as_feature_all_zero():

    A = np.array([[1, 1, 0],
                  [1, 2, 0],
                  [2, 3, 0],
                  [3, 4, 0]])
    #                    \____ all feature values zero

    b = np.array([0, 1, 2, 2])

    output = ols(A, b)
    assert output is not None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
