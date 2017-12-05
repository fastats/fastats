
from numpy.linalg import eigh

from fastats.scaling import demean


def pca(data, components=4):
    """
    Principal Component Analysis, returning
    the transformed data.

    This does not scale the data.

    Example
    -------

    >>> import numpy as np
    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> pca(x, components=2)
    array([[  5.19615242e+00,  -4.44089210e-16],
           [  0.00000000e+00,   0.00000000e+00],
           [ -5.19615242e+00,   4.44089210e-16]])
    >>> pca(x, components=1)
    array([[ 5.19615242],
           [ 0.        ],
           [-5.19615242]])
    """
    demeaned_data = demean(data)
    cov = demeaned_data.T @ demeaned_data
    _, V = eigh(cov)  # returned in increasing eigenvalue order

    V = V.T[::-1].T[:, :components]

    trans = (V.T @ demeaned_data.T).T
    return trans


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
