
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

from fastats.linear_algebra import pinv


def test_pinv_basic_sanity():
    a = np.array([
        [1, 2],
        [3, 4]
    ])

    result = pinv(a)

    expected = np.array([
        [-2.0, 1.0],
        [1.5, -0.5]
    ])

    assert_allclose(expected, result)
    assert_allclose(np.linalg.pinv(a), result)


def test_pinv_identity():
    a = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    result = pinv(a)
    assert_allclose(np.array(a, dtype='float32'), result)
    assert_allclose(np.linalg.pinv(a), result)


def test_nonsingular_2x2():
    a = np.array([
        [0, 1],
        [1, 1]
    ])

    result = pinv(a)

    expected = np.array([
        [-1.0, 1.0],
        [1.0, 0.0]
    ])

    assert_allclose(expected, result)
    assert_allclose(np.linalg.pinv(a), result, atol=1e-12)


def test_pinv_vector():
    a = np.array([[1], [3], [5]])

    result = pinv(a)
    expected = np.array([[0.02857143, 0.08571429, 0.14285714]])

    assert_allclose(expected, result)
    assert_allclose(np.linalg.pinv(a), result)


def test_pinv_nan():
    a = np.array([[1], [np.nan], [5]])

    result = pinv(a)
    expected = np.array([[np.nan, np.nan, np.nan]])

    assert_allclose(expected, result, equal_nan=True)
    assert_allclose(np.linalg.pinv(a), result, equal_nan=True)


def test_pinv_singular_raises():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    with raises(np.linalg.linalg.LinAlgError):
        _ = pinv(a)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
