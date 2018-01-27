
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose
from pytest import mark

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit
from fastats.linear_algebra import inv, matrix_minor

inv_jit = convert_to_jit(inv)
matrix_minor_jit = convert_to_jit(matrix_minor)


class MatrixInverseValidator:
    """
    This is a mixin class which tests both
    the raw Python and the jit-compiled
    version of the `inv()` function.
    """
    A, A_inv = None, None

    def setUp(self):
        self._A = np.array(self.A)

    def test_inv_outputs_numpy(self):
        A_inv = inv(self._A)
        assert A_inv.tolist() == self.A_inv

        I = np.eye(self._A.shape[0])
        assert_allclose(self.A @ A_inv, I, atol=1e-10)

    def test_inv_outputs_numba(self):
        A_inv = inv_jit(self._A)
        assert A_inv.tolist() == self.A_inv

        I = np.eye(self._A.shape[0])
        assert_allclose(self.A @ A_inv, I, atol=1e-10)


class MathworldsInv2x2Test(MatrixInverseValidator, TestCase):
    """
    This test is an example 2x2 matrix inverse from
    http://www.mathwords.com/i/inverse_of_a_matrix.htm
    """
    A = [[4, 3],
         [3, 2]]

    A_inv = [[-2, 3],
             [3, -4]]


class MathworldsInv3x3Test(MatrixInverseValidator, TestCase):
    """
    This test is an example 3x3 matrix inverse from
    http://www.mathwords.com/i/inverse_of_a_matrix.htm
    """
    A = [[1, 2, 3],
         [0, 4, 5],
         [1, 0, 6]]

    A_inv = (np.array([[24, -12, -2],
                       [5, 3, -5],
                       [-4, 2, 4]]) * 1 / 22).tolist()


class MathscentreInv3x3Test(MatrixInverseValidator, TestCase):
    """
    This test is an example 3x3 matrix inverse from
    http://www.mathcentre.ac.uk/resources/uploaded/sigma-matrices11-2009-1.pdf
    """
    A = [[7, 2, 1],
         [0, 3, -1],
         [-3, 4, -2]]

    A_inv = [[-2, 8, -5],
             [3, -11, 7],
             [9, -34, 21]]


class ImperialInv3x3Test(MatrixInverseValidator, TestCase):
    """
    This test is an example 3x3 matrix inverse from
    http://wwwf.imperial.ac.uk/metric/metric_public/matrices/inverses/inverses2.html
    """
    A = [[0, -3, -2],
         [1, -4, -2],
         [-3, 4, 1]]

    A_inv = [[4, -5, -2],
             [5, -6, -2],
             [-8, 9, 3]]


def test_hilbert_inv_5x5():
    """
    This test is an example 5x5 matrix inverse from
    http://mathfaculty.fullerton.edu/mathews/n2003/Web/InverseMatrixMod/Links/MatrixInverseMod_lnk_2.html
    """
    hilbert = np.empty((5, 5))

    for i in range(5):
        for j in range(5):
            hilbert[i][j] = 1 / (1 + i + j)

    hilbert_inv = np.array([[25, -300, 1050, -1400, 630],
                            [-300, 4800, -18900, 26880, -12600],
                            [1050, -18900, 79380, -117600, 56700],
                            [-1400, 26880, -117600, 179200, -88200],
                            [630, -12600, 56700, -88200, 44100]])

    for fn in inv, inv_jit:
        output = fn(hilbert)
        assert np.allclose(hilbert_inv, output)
        assert np.allclose(hilbert @ output, np.eye(5))


def test_inv_5x5_numpy():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15, 4, 16, 5],
                  [6, 11, 7, 9, 17],
                  [18, 19, 2, 20, 21],
                  [22, 23, 24, 25, 26]])

    A_inv = np.linalg.inv(A)

    for fn in inv, inv_jit:
        output = fn(A)
        assert np.allclose(A_inv, output)
        assert np.allclose(A @ output, np.eye(5))


@mark.parametrize('n', range(2, 10))
def test_inv_basic_sanity(n):
    """
    Note the degradation in run times for n > 5
    """
    scalar = 4
    A = np.eye(n) * scalar
    A_inv = np.eye(n) * 1 / scalar

    output = inv_jit(A)
    assert np.allclose(A_inv, output)
    assert np.allclose(A @ output, np.eye(n))


def test_matrix_minor():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15,  4, 16,  5],
                  [6, 11,  7,  9, 17]])  # <- eliminate this row (idx = 2)
    #                  \
    #                   eliminate this column (idx = 1)

    for fn in matrix_minor, matrix_minor_jit:
        output = fn(A, 2, 1)
        expected = np.array([[3, 14, 10, 12],
                             [8,  4, 16, 5]])

        assert np.allclose(output, expected)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
