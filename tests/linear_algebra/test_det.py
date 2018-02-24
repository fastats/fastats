
import importlib
import sys

import mock
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


def determinant_mathworks_test(det_fn):
    """
    3x3 matrix determinant example, taken from:
    https://uk.mathworks.com/help/matlab/ref/det.html
    """
    A = np.array([[1, -2, 4], [-5, 2, 0], [1, 0, 3]])
    assert det_fn(A) == -32
    assert det_fn(A) == approx(np.linalg.det(A))


def test_determinant_mathworks():
    determinant_mathworks_test(det)


def test_det_1x1():
    # 1x1 matrix - special case
    A = np.array([5]).reshape(1, 1)
    output = det(A)
    assert output == 5


def test_pure_python_det():

    with mock.patch('fastats.core.ast_transforms.convert_to_jit.convert_to_jit', lambda x: x):
        module_name = 'fastats.linear_algebra.det'
        importlib.reload(sys.modules[module_name])
        mod = importlib.import_module(module_name)
        det = getattr(mod, 'det')
        determinant_mathworks_test(det)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
