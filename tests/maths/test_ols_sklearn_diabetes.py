from unittest import TestCase

import statsmodels.api as sm
from sklearn import datasets

import numpy as np

from pytest import approx

from fastats.maths import ols, ols_qr
from fastats.maths.ols import (add_intercept, r_squared, sum_of_squared_residuals,
                               fitted_values, residuals, adjusted_r_squared,
                               standard_error, mean_standard_error_residuals,
                               t_statistic)


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


def test_add_intercept():

    A = np.arange(21).reshape(7, 3)
    output = add_intercept(A)
    assert output.shape == (7, 4)

    expected = sm.add_constant(A)
    assert np.allclose(output, expected)


def get_data_and_fit_statsmodel():
    data_set = datasets.load_diabetes()
    A, b = data_set.data, data_set.target
    A = add_intercept(A)  # Note addition of intercept
    model = sm.OLS(b, sm.add_constant(A)).fit()
    return A, b, model


def test_r_squared():
    A, b, model = get_data_and_fit_statsmodel()
    expected = model.rsquared
    output = r_squared(A, b)
    assert output == approx(expected)


def test_adjusted_r_squared():
    A, b, model = get_data_and_fit_statsmodel()
    expected = model.rsquared_adj
    output = adjusted_r_squared(A, b)
    assert output == approx(expected)


def test_sum_of_squared_residuals():
    A, b, model = get_data_and_fit_statsmodel()
    expected = model.ssr
    output = sum_of_squared_residuals(A, b)
    assert output == approx(expected)


def test_fitted_values():
    A, b, model = get_data_and_fit_statsmodel()
    expected = model.fittedvalues
    output = fitted_values(A, b)
    assert np.allclose(output, expected)


def test_residuals():
    A, b, model = get_data_and_fit_statsmodel()
    expected = model.resid
    output = residuals(A, b)
    assert np.allclose(output, expected)


def test_standard_error():
    A, b, model = get_data_and_fit_statsmodel()
    expected = model.bse
    output = standard_error(A, b)
    assert np.allclose(output, expected)


def test_mean_standard_error_residuals():
    A, b, model = get_data_and_fit_statsmodel()
    output = mean_standard_error_residuals(A, b)
    expected = model.mse_resid
    assert np.allclose(output, expected)


def test_t_statistic():
    A, b, model = get_data_and_fit_statsmodel()
    expected = model.tvalues
    output = t_statistic(A, b)
    assert np.allclose(output, expected)


if __name__ == '__main__':
    import pytest
    pytest.main()
