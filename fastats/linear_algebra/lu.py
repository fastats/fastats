
from numpy import sum as nsum
import numpy as np

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit


def lu_inplace(A, L, U):
    """
    LU Decomposition

    Takes a square `numpy.array` A and decomposes
    it into the lower and upper matrices `L` and `U`.

    This scales as O(n^3).

    This is a special variant that takes the `L` and `U`
    arrays pre-initialised to zero. If you're iterating
    over a lot of matrices, this can save you from
    allocating new `L` and `U` matrices on each
    iteration. From profiling, this saves ~10-20%
    of the runtime for small matrices.

    Example
    -------
    This is the example from wikipedia:
    https://en.wikipedia.org/wiki/LU_decomposition

    >>> import numpy as np
    >>> A = np.array([[4, 3], [6, 3]], dtype=np.float32)
    >>> L, U = np.zeros_like(A), np.zeros_like(A)
    >>> lu_inplace(A, L, U)
    >>> L
    array([[ 1. , -0. ],
           [ 1.5,  1. ]], dtype=float32)
    >>> U
    array([[ 4. ,  3. ],
           [ 0. , -1.5]], dtype=float32)
    """
    assert A.shape[0] == A.shape[1]
    assert L.shape[0] == L.shape[1] == A.shape[0]
    assert U.shape[0] == U.shape[1] == A.shape[0]

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


lu_inplace_jit = convert_to_jit(lu_inplace)


def lu(A):
    """
    This performs LU Decomposition on `A`.

    This takes a square matrix `A`.

    This scales as O(n^3).

    This allocates `L` and `U` on each call.

    Example
    -------
    This is the example from wikipedia:
    https://en.wikipedia.org/wiki/LU_decomposition

    >>> import numpy as np
    >>> A = np.array([[4, 3], [6, 3]], dtype=np.float32)
    >>> L, U = lu(A)
    >>> L
    array([[ 1. , -0. ],
           [ 1.5,  1. ]], dtype=float32)
    >>> U
    array([[ 4. ,  3. ],
           [ 0. , -1.5]], dtype=float32)
    """
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    lu_inplace_jit(A, L, U)
    return L, U


def lu_compact(A):
    """
    This performs LU Decomposition on `A`
    """
    assert A.shape[0] == A.shape[1]

    U = A.astype(np.float64)
    n = A.shape[0]
    L = np.eye(n).astype(np.float64)

    for k in range(n - 1):
        for j in range(k + 1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:n] -= L[j, k] * U[k, k:n]

    return L, U


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
