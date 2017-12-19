
import sys

from hypothesis import given
from hypothesis.strategies import integers

from numba import jit
import numpy as np


# Parallel not supported on 32-bit Windows
parallel = not (sys.platform == 'win32')


def get(n):
    return np.ones((n, 1), dtype=np.float64)


get_jit = jit(nopython=True, parallel=parallel)(get)


@given(integers(min_value=10, max_value=100000))
def test_all_ones(x):
    """
    We found one of the scaling tests failing on
    OS X with numba 0.35, but it passed on other
    platforms, and passed consistently with numba
    0.36.

    The issue appears to be the same as numba#2609
    https://github.com/numba/numba/issues/2609
    so we've taken the minimal repro from that issue
    and are using it as a unit-test here.
    """
    result = get_jit(x)
    expected = get(x)
    assert np.allclose(expected, result)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
