
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

from fastats.linear_algebra import matrix_minor


def test_matrix_minor():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15, 4, 16, 5],
                  [6, 11, 7, 9, 17]])  # <- eliminate this row (idx = 2)
    #                  \
    #                   eliminate this column (idx = 1)

    output = matrix_minor(A, 2, 1)
    expected = np.array([[3, 14, 10, 12],
                         [8,  4, 16, 5]])

    assert_allclose(output, expected)


def test_matrix_minor_2x2():

    A = np.array([[3, 13],
                  [8, 15]])

    output = matrix_minor(A, 1, 1)
    expected = np.array([[3]])
    assert_allclose(output, expected)


def test_matrix_minor_perimeter():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15, 4, 16, 5],
                  [6, 11, 7, 9, 17]])

    output = matrix_minor(A, 2, 4)
    expected = np.array([[3, 13, 14, 10],
                         [8, 15, 4, 16]])

    assert_allclose(output, expected)


def matrix_minor_interior_test():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15, 4, 16, 5],
                  [6, 11, 7, 9, 17]])

    output = matrix_minor(A, 1, 3)
    expected = np.array([[3, 13, 14, 12],
                         [6, 11, 7, 17]])

    assert_allclose(output, expected)


def test_matrix_minor_raises_if_idx_out_of_bounds():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15, 4, 16, 5],
                  [6, 11, 7, 9, 17]])

    with raises(AssertionError):
        _ = matrix_minor(A, 0, 5)  # 5 exceeds index of final column

    with raises(AssertionError):
        _ = matrix_minor(A, 3, 0)  # 3 exceeds index of final row


def test_matrix_minor_raises_if_array_not_at_least_2x2():

    A = np.array([3, 13, 14, 10, 12]).reshape(1, 5)

    with raises(AssertionError):
        _ = matrix_minor(A, 0, 2)

    A = np.array([3, 13, 14, 10, 12]).reshape(5, 1)

    with raises(AssertionError):
        _ = matrix_minor(A, 2, 0)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
