
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose
from pytest import approx

from fastats.linear_algebra import lu, lu_inplace
from fastats.core.ast_transforms.convert_to_jit import convert_to_jit


lu_jit = convert_to_jit(lu)


class LUDecompValidator:
    """
    This is a mixin class which tests both
    the raw python and the jit-compiled
    version of the `lu()` function.
    """
    A, L, U = None, None, None

    def setUp(self):
        self._A = np.array(self.A)

    def test_lu_outputs_numpy(self):
        L = np.zeros_like(self._A)
        U = np.zeros_like(self._A)

        lu_inplace(self._A, L, U)

        assert L.tolist() == self.L
        assert U.tolist() == self.U
        assert_allclose(L @ U, self._A)

        raw_L, raw_U = lu(self._A)

        assert raw_L.tolist() == self.L
        assert raw_U.tolist() == self.U

        assert_allclose(raw_L @ raw_U, self._A)

    def test_lu_outputs_numba(self):

        L, U = lu_jit(self._A)

        assert L.tolist() == self.L
        assert U.tolist() == self.U

        assert_allclose(L @ U, self._A)


class SotonLUTests(LUDecompValidator, TestCase):
    """
    This tests the example from
    http://www.personal.soton.ac.uk/jav/soton/HELM/workbooks/workbook_30/30_3_lu_decomposition.pdf
    stored in fastats/literature/soton_lu_decomp.pdf
    """
    A = [[1, 2, 4],
         [3, 8, 14],
         [2, 6, 13]]

    L = [[1, 0, 0],
         [3, 1, 0],
         [2, 1, 1]]

    U = [[1, 2, 4],
         [0, 2, 2],
         [0, 0, 3]]


class OhioFacultyLUTests(LUDecompValidator, TestCase):
    """
    This tests the example from
    http://www.ohiouniversityfaculty.com/youngt/IntNumMeth/lecture12.pdf
    stored in fastats/literature/ohio_faculty_lu_decomp.pdf
    """
    A = [[1, -2, 3],
         [2, -5, 12],
         [0, 2, -10]]

    L = [[1, 0, 0],
         [2, 1, 0],
         [0, -2, 1]]

    U = [[1, -2, 3],
         [0, -1, 6],
         [0, 0, 2]]


class IITLUTests(LUDecompValidator, TestCase):
    """
    This tests the example from
    http://www.math.iit.edu/~fass/477577_Chapter_7.pdf
    stored in fastats/literature/IIT_LU_decomposition.pdf
    """
    A = [[1, 1, 1],
         [2, 3, 5],
         [4, 6, 8]]

    L = [[1, 0, 0],
         [2, 1, 0],
         [4, 2, 1]]

    U = [[1, 1, 1],
         [0, 1, 3],
         [0, 0, -2]]


class NorthwesternLUTests(LUDecompValidator, TestCase):
    """
    This tests the 4x4 example from
    http://www.math.northwestern.edu/~clark/285/2006-07/handouts/lu-factor.pdf
    stored in fastats/literature/northwestern_lu_decomp.pdf
    """
    A = [[3, -7, -2, 2],
         [-3, 5, 1, 0],
         [6, -4, 0, -5],
         [-9, 5, -5, 12]]

    L = [[1, 0, 0, 0],
         [-1, 1, 0, 0],
         [2, -5, 1, 0],
         [-3, 8, 3, 1]]

    U = [[3, -7, -2, 2],
         [0, -2, -1, 2],
         [0, 0, -1, 1],
         [0, 0, 0, -1]]


class UNCCLUTests(LUDecompValidator, TestCase):
    """
    This tests the 4x4 floating-point example from
    https://webpages.uncc.edu/krs/courses/4133-5133/lectures/lu2.pdf
    stored in fastats/literature/uncc_lu_decomp.pdf
    """
    A = [[4., 12., 8., 4.],
         [1., 7., 18., 9.],
         [2., 9., 20., 20.],
         [3., 11., 15., 14.]]

    L = [[1, 0, 0, 0],
         [0.25, 1.0, 0.0, 0.0],
         [0.5, 0.75, 1.0, 0.0],
         [0.75, 0.5, 0.25, 1.0],
         ]

    U = [[4.0, 12.0, 8.0, 4.0],
         [0.0, 4.0, 16.0, 8.0],
         [0.0, 0.0, 4.0, 12.0],
         [0.0, 0.0, 0.0, 4.0]]


class UNCCTridiagonalLUTests(LUDecompValidator, TestCase):
    """
    This tests the 4x4 tridiagonal example from
    https://webpages.uncc.edu/krs/courses/4133-5133/lectures/lu2.pdf
    stored in fastats/literature/uncc_lu_decomp.pdf
    """
    A = [[2.0, -1.0, 0.0, 0.0],
         [-1.0, 2.0, -1.0, 0.0],
         [0.0, -1.0, 2.0, -1.0],
         [0.0, 0.0, -1.0, 2.0]]

    L = [[1.0, 0.0, 0.0, 0.0],
         [-0.5, 1.0, 0.0, 0.0],
         [0.0, approx(-2./3.), 1.0, 0.0],
         [0.0, 0.0, approx(-0.75), 1.0]]

    U = [[2.0, -1.0, 0.0, 0.0],
         [0.0, 1.5, -1.0, 0.0],
         [0.0, 0.0, approx(4./3.), -1.0],
         [0.0, 0.0, 0.0, approx(5./4.)]]


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
