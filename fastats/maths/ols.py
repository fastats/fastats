from numpy import diag, sqrt, hstack, ones, eye
from numpy.linalg import inv
from scipy.linalg import qr, solve_triangular


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
    return inv(A.T @ A) @ A.T @ b


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


def add_intercept(A):
    """
    Adds an intercept (column of ones) to a supplied array A
    such that the intercept is the first (leftmost) column
    """
    n = A.shape[0]
    intercept = ones(n).reshape(n, 1)
    return hstack([intercept, A])


def _hat(A):
    """
    The 'hat' matrix for an array A
    """
    return A @ inv(A.T @ A) @ A.T


def _m_matrix(A):
    """
    The 'hat' matrix for a vector of ones whose size is
    equal to the first dimension of the supplied array A
    """
    n = A.shape[0]
    l = ones(n).reshape((n, 1))
    return _hat(l)


def sum_of_squared_residuals(A, b):
    """
    The sum of squared residuals
    """
    n = A.shape[0]
    I = eye(n)
    H = _hat(A)
    return b.T @ (I - H) @ b


def total_sum_of_squares(A, b):
    """
    The total sum of squares
    """
    n = A.shape[0]
    I = eye(n)
    M = _m_matrix(A)
    return b.T @ (I - M) @ b


def r_squared(A, b):
    """
    The r-squared value (a.k.a. coefficient of determination)
    """
    ssr = sum_of_squared_residuals(A, b)
    sst = total_sum_of_squares(A, b)
    return 1.0 - ssr / sst


def adjusted_r_squared(A, b):
    """
    The adjusted r-squared value
    """
    ssr = sum_of_squared_residuals(A, b)
    sst = total_sum_of_squares(A, b)
    n, k = A.shape
    return 1.0 - (ssr / (n - k) / (sst / (n - 1)))


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


if __name__ == '__main__':
    import pytest
    pytest.main()

