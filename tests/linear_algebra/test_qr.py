
from unittest import TestCase

import numpy as np
from numpy.testing import assert_allclose

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit
from fastats.linear_algebra import qr, qr_classical_gram_schmidt


qr_jit = convert_to_jit(qr)
qr_qr_classical_gram_schmidt_jit = convert_to_jit(qr_classical_gram_schmidt)


class QRTestMixin:

    @staticmethod
    def assert_orthonormal(Q):
        n = Q.shape[1]
        assert_allclose(Q.T @ Q, np.eye(n), atol=1e-10)

    @staticmethod
    def check_versus_expectations(Q, Q_expected, R, R_expected, A):
        assert_allclose(Q, Q_expected)
        assert_allclose(R, R_expected)
        assert_allclose(Q @ R, A)

    def test_ucla(self):
        # http://www.seas.ucla.edu/~vandenbe/133A/lectures/qr.pdf
        A = np.array([[-1, -1, 1],
                      [1, 3, 3],
                      [-1, -1, 5],
                      [1, 3, 7]])

        Q_expected = np.array([[-0.5, 0.5, -0.5],
                               [0.5, 0.5, -0.5],
                               [-0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5]])

        R_expected = np.array([[2, 4, 2],
                               [0, 2, 8],
                               [0, 0, 4]])

        Q, R = self.fn(A)
        self.check_versus_expectations(Q, Q_expected, R, R_expected, A)
        self.assert_orthonormal(Q)

    def test_wikipedia(self):
        # https://en.wikipedia.org/wiki/QR_decomposition
        A = np.array([[12, -51, 4],
                      [6, 167, -68],
                      [-4, 24, -41]])

        Q_expected = np.array([[6/7, -69/175, -58/175],
                               [3/7, 158/175, 6/175],
                               [-2/7, 6/35, -33/35]])

        R_expected = np.array([[14, 21, -14],
                               [0, 175, -70],
                               [0, 0, 35]])

        Q, R = self.fn(A)
        self.check_versus_expectations(Q, Q_expected, R, R_expected, A)
        self.assert_orthonormal(Q)


class QRTests(QRTestMixin, TestCase):

    def setUp(self):
        self.fn = qr


class QRClassicalGSTests(QRTestMixin, TestCase):

    def setUp(self):
        self.fn = qr_classical_gram_schmidt


class QRJitTests(QRTestMixin, TestCase):

    def setUp(self):
        self.fn = qr_jit


class QRClassicalGSJitTests(QRTestMixin, TestCase):

    def setUp(self):
        self.fn = qr_qr_classical_gram_schmidt_jit


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
