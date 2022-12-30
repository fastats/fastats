
import numpy as np


def matrix_minor(A, remove_row_idx, remove_col_idx):
    """
    Returns a minor matrix, cut down from A by removing
    one of its rows and one of its columns.

    >>> A = np.array([[4, 3, 5], [3, 2, 6], [3, 2, 7]])
    >>> A_inv = matrix_minor(A, 2, 2)
    >>> A_inv
    array([[4., 3.],
           [3., 2.]])
    """
    m, n = A.shape
    assert m > 1 and n > 1
    assert remove_row_idx <= m - 1, 'Row index out of bounds'
    assert remove_col_idx <= n - 1, 'Column index out of bounds'

    res = np.empty(shape=(m - 1, n - 1))
    retained_row_idx = -1
    retained_col_indices = np.arange(n) != remove_col_idx

    for x in range(m):
        if x != remove_row_idx:
            retained_row_idx += 1
            res[retained_row_idx, :] = A[x, :][retained_col_indices]

    return res


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
