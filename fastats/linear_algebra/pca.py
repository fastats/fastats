
from numpy.linalg import eigh

from fastats.scaling import demean


def pca(data, components=4):
    """
    Principal Component Analysis, returning
    the transformed data.

    This does not scale the data.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.array([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9]
    ... ])
    >>> np.abs(pca(x, components=1))
    array([[5.19615242],
           [0.        ],
           [5.19615242]])
    """
    demeaned_data = demean(data)
    cov = demeaned_data.T @ demeaned_data
    # eigh returns ordered eigenvalues
    _, V = eigh(cov)

    V = V.T[::-1].T[:, :components]

    trans = (V.T @ demeaned_data.T).T
    return trans


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
