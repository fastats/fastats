
from math import sqrt
from unittest import TestCase

from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
from scipy import linalg

from pytest import approx

from fastats.maths import ols, ols_qr


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


if __name__ == '__main__':
    import pytest
    pytest.main()
