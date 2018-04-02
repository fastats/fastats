
import numpy as np
from pytest import approx

from fastats.linear_algebra import det


def test_determinant_wikihow():
    """
    2x2 matrix determinant example, taken from:
    https://www.wikihow.com/Find-the-Determinant-of-a-2x2-Matrix
    """
    A = np.array([[9, 8], [-7, 6]])
    assert det(A) == 110
    assert det(A) == approx(np.linalg.det(A))


def test_determinant_wikipedia():
    """
    3x3 matrix determinant example, taken from:
    https://en.wikipedia.org/wiki/Determinant
    """
    A = np.array([[-2, 2, -3], [-1, 1, 3], [2, 0, -1]])
    assert det(A) == 18
    assert det(A) == approx(np.linalg.det(A))


def determinant_mathworks_test():
    """
    3x3 matrix determinant example, taken from:
    https://uk.mathworks.com/help/matlab/ref/det.html
    """
    A = np.array([[1, -2, 4], [-5, 2, 0], [1, 0, 3]])
    assert det(A) == -32
    assert det(A) == approx(np.linalg.det(A))


def test_determinant_1x1():
    """
    1x1 matrix special case where det returns the single
    matrix element
    """
    A = np.array([5]).reshape(1, 1)
    output = det(A)
    assert output == 5


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
