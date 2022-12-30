
from fastats.linear_algebra.matrix_minor import matrix_minor


def det(A):
    """
    Returns the determinant of A.

    >>> A = np.array([[4, 3], [3, 2]], dtype=np.float32)
    >>> det(A)
    -1.0
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
            determinant += ((-1) ** j) * A[0][j] * det(matrix_minor(A, 0, j))

    return determinant


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
