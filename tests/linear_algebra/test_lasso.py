
import numpy as np
from numpy.testing import assert_allclose
from pytest import mark, fixture
from scipy.stats import ortho_group
from sklearn.linear_model import Lasso, LinearRegression

from fastats.linear_algebra import lasso_orthonormal
from fastats.linear_algebra.lasso import soft_threshold


@fixture
def data():
    rng = np.random.RandomState(0)

    # generate a 50 by 100 array of orthonormal features
    features = ortho_group.rvs(100, random_state=0)[:, :50]
    assert_orthonormal(features)

    # generate x_true; set the first 25 elements to zero
    m, n = features.shape
    x_true = rng.rand(n)
    x_true[:25] = 0

    # generate some noise and compute targets as sum of x_true and noise
    noise = rng.randn(m) * np.sqrt(0.001)
    targets = features @ x_true + noise

    return features, targets


@mark.parametrize('lambda_', np.logspace(-5, -3, 10), ids='lambda_{0:.3E}'.format)
def test_lasso_orthonormal_vs_sklearn(lambda_, data):
    features, targets = data

    # generate lasso regression coefficients and check vs sklearn
    output = lasso_orthonormal(features, targets, lambda_)
    expected = Lasso(alpha=lambda_, fit_intercept=False).fit(features, targets).coef_
    assert_allclose(expected, output)


def test_lasso_orthonormal_lambda_zero(data):
    features, targets = data
    lambda_ = 0

    # no L1 penalty so should behave like a linear regression
    output = lasso_orthonormal(features, targets, lambda_)
    expected = LinearRegression(fit_intercept=False).fit(features, targets).coef_
    assert_allclose(expected, output)


def assert_orthonormal(features):
    n = features.shape[1]
    assert_allclose(features.T @ features, np.eye(n), atol=1e-5)


def test_soft_threshold():
    lambda_ = 2

    # abs values above threshold are moved closer to 0 by threshold amount
    assert soft_threshold(5, lambda_) == 3    # -> (5 - 2)
    assert soft_threshold(-5, lambda_) == -3  # -> (-5 + 2)

    # abs values equal to or less than threshold are set to 0
    assert soft_threshold(2, lambda_) == 0
    assert soft_threshold(1, lambda_) == 0
    assert soft_threshold(-1, lambda_) == 0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
