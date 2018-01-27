
import numpy as np

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit


def matrix_minor(A, remove_row_idx, remove_col_idx):
    """
    Returns a square matrix, cut down from A by removing
    one of its rows and one of its columns.
    """
    m = A.shape[0] - 1
    n = A.shape[1] - 1
    res = np.empty(shape=(m, n))

    retained_row_idx = -1
    retained_col_indices = np.arange(n + 1) != remove_col_idx

    for x in range(m + 1):
        if x != remove_row_idx:
            retained_row_idx += 1
            res[retained_row_idx, :] = A[x, :][retained_col_indices]

    return res


matrix_minor_jit = convert_to_jit(matrix_minor)


def det(A):
    """
    Returns the determinant of A.
    """
    m, n = A.shape
    assert m == n

    if m == 1:
        determinant = A[0][0]
    elif m == 2:
        determinant = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    else:
        determinant = 0
        for j in range(m):
            determinant += ((-1) ** j) * A[0][j] * det_jit(matrix_minor_jit(A, 0, j))

    return determinant


det_jit = convert_to_jit(det)


def inv(A):
    """
    Returns the inverse of A using the adjoint method.

    >>> import numpy as np
    >>> A = np.array([[4, 3], [3, 2]])
    >>> A_inv = inv(A)
    >>> A_inv
    array([[-2.,  3.],
           [ 3., -4.]])
    """
    m, n = A.shape
    co_factors = np.empty_like(A, dtype=np.float64)

    for r in range(n):
        for c in range(m):
            minor = matrix_minor_jit(A, r, c)
            co_factors[r, c] = ((-1) ** (r + c)) * det_jit(minor)

    adjoint = co_factors.T
    return adjoint / det_jit(A)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
