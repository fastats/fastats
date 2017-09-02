
import numpy as np
from pytest import approx

from fastats.maths import sum_sq_dev


def test_basic_sanity():
    a = np.array([1, 2, 3])

    # 1^2 + 1^2
    assert sum_sq_dev(a) == approx(2.0)

    b = np.array([1, 2, 3, 4, 5])

    # 1^2 + 1^2 + 2^2 + 2^2
    assert sum_sq_dev(b) == approx(10.0)

    c = np.array([0.5, 1.0, 1.5])

    # 0.5^2 + 0.5^2
    assert sum_sq_dev(c) == approx(0.5)


def test_nan_values():
    a = np.array([1, 2, np.nan])

    # 0.5^2 + 0.5^2
    assert not np.isfinite(sum_sq_dev(a))

    b = np.array([np.nan, 2, 3])

    assert not np.isfinite(sum_sq_dev(b))


if __name__ == '__main__':
    import pytest
    pytest.main()
