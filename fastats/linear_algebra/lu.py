
from numpy import sum as nsum


def lu(A, L, U):
    """
    LU Decomposition

    Takes a square `numpy.array` A and decomposes
    it into the lower and upper matrices `L` and `U`.

    This scales as O(n^3).

    You need to pass in the `L` and `U` arrays
    pre-initialised to zero.

    Example
    -------
    This is the example from wikipedia:
    https://en.wikipedia.org/wiki/LU_decomposition

    >>> import numpy as np
    >>> A = np.array([[4, 3], [6, 3]], dtype=np.float32)
    >>> L, U = np.zeros_like(A), np.zeros_like(A)
    >>> lu(A, L, U)
    >>> L
    array([[ 1. , -0. ],
           [ 1.5,  1. ]], dtype=float32)
    >>> U
    array([[ 4. ,  3. ],
           [ 0. , -1.5]], dtype=float32)
    """
    size = len(A)

    for i in range(size):
        for k in range(size):
            total = nsum(L[i, 0:i] * U[0:i, k])
            U[i, k] = A[i, k] - total

        for k in range(size):
            if i == k:
                L[i, i] = 1.0
            else:
                total = nsum(L[k, 0:i] * U[0:i, i])
                L[k, i] = (A[k, i] - total) / U[i, i]


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
