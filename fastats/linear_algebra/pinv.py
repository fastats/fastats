
import numpy as np


def pinv(A):
    """
    Pseudo-inverse.

    `A.T @ A` must be invertible (obviously),
    which means that `A` must generally be
    positive semidefinite.

    >>> import numpy as np
    >>> arr = np.array([
    ...     [1, 3],
    ...     [2, 4]
    ... ])
    >>> pinv(arr)
    array([[-2. ,  1.5],
           [ 1. , -0.5]])
    """
    return np.linalg.inv(A.T @ A) @ A.T


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
