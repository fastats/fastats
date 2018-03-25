
import numpy as np


def qr_classical_gram_schmidt(A):
    """
    Returns the QR decomposition of matrix A using
    the modified Gram-Schmidt method.
    """
    A = A.copy().astype(np.float64)
    n = A.shape[1]
    Q = np.zeros_like(A, dtype=np.float64)
    R = np.zeros(shape=(n, n), dtype=np.float64)

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - (R[i, j] * Q[:, i])
        R[j, j] = (v.T @ v) ** 0.5
        Q[:, j] = v / R[j, j]

    return Q, R


def qr(A):
    """
    Returns the QR decomposition of matrix A using
    the modified Gram-Schmidt method (single matrix
    projection rather than sequence of vector
    projections).

    Exhibits superior numerical stability versus
    classical method.
    """
    V = A.copy().astype(np.float64)
    n = A.shape[1]
    Q = np.zeros_like(A, dtype=np.float64)
    R = np.zeros(shape=(n, n), dtype=np.float64)

    for i in range(n):
        v = V[:, i]
        R[i, i] = (v.T @ v) ** 0.5
        Q[:, i] = v / R[i, i]
        for j in range(i, n):
            R[i, j] = Q[:, i] @ V[:, j]
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

    return Q, R
