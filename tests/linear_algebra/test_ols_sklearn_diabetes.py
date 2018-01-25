
from unittest import TestCase

import numpy as np
import statsmodels.api as sm
from pytest import approx
from sklearn import datasets

from fastats.linear_algebra import (
    ols, ols_cholesky, ols_qr, ols_svd,
    add_intercept, adjusted_r_squared,
    adjusted_r_squared_no_intercept, fitted_values,
    mean_standard_error_residuals, r_squared,
    r_squared_no_intercept, residuals, standard_error,
    sum_of_squared_residuals, t_statistic, f_statistic,
    f_statistic_no_intercept
)


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


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
