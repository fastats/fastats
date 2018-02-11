
import numpy as np

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit
from fastats.linear_algebra.det import det
from fastats.linear_algebra.matrix_minor import matrix_minor


@convert_to_jit
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

    for i in range(n):
        for j in range(m):
            minor = matrix_minor(A, i, j)
            co_factors[i, j] = ((-1) ** (i + j)) * det(minor)

    adjoint = co_factors.T
    return adjoint / det(A)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
