
from numpy.linalg import inv
from scipy import linalg


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


def ols_qr(A,b):
    """
    Ordinary Least-Squares Regression
    Coefficients Estimation.

    This is the QR Factorization solution
    to OLS; Ax = b, A = QR, QRx = b,
    therefore Rx = Q.T * b
    """
    Q, R = linalg.qr(A, mode='economic')
    return linalg.solve_triangular(R, Q.T @ b)


if __name__ == '__main__':
    import pytest
    pytest.main()
