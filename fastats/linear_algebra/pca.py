
import numpy as np
from numpy import cov
from scipy.linalg import eigh


def pca(data, components=4):
    """
    Principal Component Analysis using raw numpy
    and scipy calls, returning the transformed data.

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
    mu = data.mean(axis=0)

    # Have a think about this - this effectively copies
    # the input just to demean. If we don't, we modify
    # the input.
    x = data - mu
    r = cov(x, rowvar=False)
    S, V = eigh(r)

    idx = np.argsort(S)[::-1]
    V = V[:, idx]

    V = V[:, :components]
    trans = np.dot(V.T, x.T).T
    return trans


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
