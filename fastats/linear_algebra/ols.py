
import numpy as np
from numpy import diag, sqrt, hstack, ones, eye
from numpy.linalg import inv
from scipy.linalg import qr, solve_triangular, cholesky, svd


def ols(A, b):
    """
    Ordinary Least-Squares Regression
    Coefficients Estimation.

    This is the linear algebra solution to
    OLS; minimising |Ax-b| means we want the
    values where the perpendicular is zero.
    This means we want columnwise-multiplication
    with A, giving A.T(Ax-b), therefore
    0 = A.T*Ax - A.T*b, therefore
    x = A.T*b / A.T*A. QED.
    """
    return np.linalg.inv(A.T @ A) @ A.T @ b


def ols_qr(A, b):
    """
    Ordinary Least-Squares Regression
    Coefficients Estimation.

    This is the QR Factorization solution
    to OLS; Ax = b, A = QR, QRx = b,
    therefore Rx = Q.T * b
    """
    Q, R = qr(A, mode='economic')
    return solve_triangular(R, Q.T @ b)


def ols_cholesky(A, b):
    """
    Ordinary Least-Squares Regression Coefficients
    Estimation.

    If (A.T @ A) @ x = A.T @ b and A is full rank
    then there exists an upper triangular matrix
    R such that:

    (R.T @ R) @ x = A.T @ b
    R.T @ w = A.T @ b
    R @ x = w

    Find R using Cholesky decomposition.
    """
    R = cholesky(A.T @ A)
    w = solve_triangular(R, A.T @ b, trans='T')
    return solve_triangular(R, w)


def ols_svd(A, b):
    """
    Ordinary Least-Squares Regression Coefficients
    Estimation.

    A @ x = b
    A = U @ Σ @ Vh (singular value decomposition of A)
    Σ @ Vh @ x = U.T @ b
    Σ @ w = U.T @ b
    x = Vh.T @ w
    """
    U, sigma, Vh = svd(A, full_matrices=False)
    w = (U.T @ b) / sigma
    return Vh.T @ w


def add_intercept(A):
    """
    Adds an intercept (column of ones) to a supplied array A
    such that the intercept is the first (leftmost) column
    """
    n = A.shape[0]
    intercept = ones(n).reshape(n, 1)
    return hstack((intercept, A))


def _hat(A):
    """
    The 'hat' matrix for an array A
    """
    return A @ np.linalg.inv(A.T @ A) @ A.T


def _m_matrix(A):
    """
    The 'hat' matrix for a vector of ones whose size is
    equal to the first dimension of the supplied array A
    """
    n = A.shape[0]
    l = np.ones(n).reshape(n, 1)
    return _hat(l)


def sum_of_squared_residuals(A, b):
    """
    The sum of squared residuals
    """
    n = A.shape[0]
    I = np.eye(n)
    H = _hat(A)
    return b.T @ (I - H) @ b


def total_sum_of_squares(A, b):
    """
    The total sum of squares
    """
    n = A.shape[0]
    I = np.eye(n)
    M = _m_matrix(A)
    return b.T @ (I - M) @ b


def r_squared(A, b):
    """
    The r-squared value (a.k.a. coefficient of determination)
    """
    ssr = sum_of_squared_residuals(A, b)
    sst = total_sum_of_squares(A, b)
    return 1.0 - ssr / sst


def r_squared_no_intercept(A, b):
    """
    The r-squared value (a.k.a. coefficient of determination) in the
    case where A has no intercept, per statsmodels - compare the
    slope-only model to a model that simply makes a constant
    prediction of 0 for all observations
    """
    fitted = fitted_values(A, b)
    return (fitted.T @ fitted) / (b.T @ b)


def adjusted_r_squared(A, b):
    """
    The adjusted r-squared value
    """
    n, k = A.shape
    return 1 - (n - 1) / (n - k) * (1 - r_squared(A, b))


def adjusted_r_squared_no_intercept(A, b):
    """
    The adjusted r-squared value in the case where no intercept term is present
    """
    n, k = A.shape
    return 1 - n / (n - k) * (1 - r_squared_no_intercept(A, b))


def fitted_values(A, b):
    """
    The predicted values for b
    """
    return _hat(A) @ b


def _residual_maker(A, b):
    """
    Returns a matrix which can be used to make residuals from b
    """
    n = A.shape[0]
    I = eye(n)
    return I - _hat(A)


def residuals(A, b):
    """
    The residuals of the model
    """
    return _residual_maker(A, b) @ b


def mean_standard_error_residuals(A, b):
    """
    Mean squared error of the residuals. The sum of squared residuals
    divided by the residual degrees of freedom.
    """
    n, k = A.shape
    ssr = sum_of_squared_residuals(A, b)
    return ssr / (n - k)


def mean_standard_error_model(A, b):
    """
    Mean squared error the model.
    """
    k = A.shape[1]

    sst = total_sum_of_squares(A, b)
    ssr = sum_of_squared_residuals(A, b)
    sse = sst - ssr

    return sse / (k - 1)


def mean_standard_error_model_no_intercept(A, b):
    """
    Mean squared error the model in the case where A has no
    intercept and the model is slope-only.
    """
    k = A.shape[1]

    fitted = fitted_values(A, b)
    sse = fitted.T @ fitted

    return sse / k


def standard_error(A, b):
    """
    The standard errors of the parameter estimates.
    """
    mse = mean_standard_error_residuals(A, b)
    C = inv(A.T @ A)
    return sqrt(diag(C) * mse)


def t_statistic(A, b):
    """
    t-statistics for the model.
    """
    betas = ols(A, b)
    se = standard_error(A, b)
    return betas / se


def f_statistic(A, b):
    """
    F-statistic of the fully specified model.
    """
    mse_model = mean_standard_error_model(A, b)
    mse_resid = mean_standard_error_residuals(A, b)
    return mse_model / mse_resid


def f_statistic_no_intercept(A, b):
    """
    F-statistic of the fully specified model in the case where
    A has no intercept (a slope-only model).
    """
    mse_model = mean_standard_error_model_no_intercept(A, b)
    mse_resid = mean_standard_error_residuals(A, b)
    return mse_model / mse_resid


def drop_missing(A, b):
    """
    Returns a filtration of A (features) and b (targets) where all
    values are not NaN, with the intention that OLS can then be
    performed on dense / 'complete' observations.

    This is analogous to the statsmodels missing='drop' mechanism.
    """
    """
    Returns a filtration of A (features) and b (targets) where all
    values are not NaN, with the intention that OLS can then be
    performed on dense / 'complete' observations.

    This is analogous to the statsmodels missing='drop' mechanism.
    """
    m = A.shape[0]
    assert m == len(b)

    A_bar = np.empty_like(A)
    b_bar = np.empty_like(b)

    ctr = 0

    for i in range(m):
        feature_row = A[i, :]
        target = b[i]
        if np.all(~np.isnan(feature_row)) and ~np.isnan(target):
            A_bar[ctr, :] = feature_row
            b_bar[ctr] = target
            ctr += 1

    return A_bar[:ctr, :], b_bar[:ctr]


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
