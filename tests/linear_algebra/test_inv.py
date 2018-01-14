
import numpy as np

from fastats.linear_algebra import inv, matrix_minor


def test_inv_mathwords_2_by_2():

    # http://www.mathwords.com/i/inverse_of_a_matrix.htm
    A = np.array([[4, 3],
                  [3, 2]])

    A_inv = np.array([[-2, 3],
                      [3, -4]])

    output = inv(A)
    assert np.allclose(A_inv, output)
    assert np.allclose(A @ output, np.eye(2))


def test_inv_mathwords_3_by_3():

    # http://www.mathwords.com/i/inverse_of_a_matrix.htm
    A = np.array([[1, 2, 3],
                  [0, 4, 5],
                  [1, 0, 6]])

    A_inv = np.array([[24, -12, -2],
                      [ 5,   3, -5],
                      [-4,   2,  4]]) * 1 / 22

    output = inv(A)
    assert np.allclose(A_inv, output)
    assert np.allclose(A @ output, np.eye(3))


def test_inv_imperial_3_by_3():

    # http: // wwwf.imperial.ac.uk / metric / metric_public / matrices / inverses / inverses2.html
    A = np.array([[0, -3, -2],
                  [1, -4, -2],
                  [-3, 4, 1]])

    A_inv = np.array([[4, -5, -2],
                      [5, -6, -2],
                      [-8, 9, 3]])

    output = inv(A)
    assert np.allclose(A_inv, output)
    assert np.allclose(A @ output, np.eye(3))


def test_inv_5_by_5_numpy():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15, 4, 16, 5],
                  [6, 11, 7, 9, 17],
                  [18, 19, 2, 20, 21],
                  [22, 23, 24, 25, 26]])

    A_inv = np.linalg.inv(A)

    output = inv(A)
    assert np.allclose(A_inv, output)
    assert np.allclose(A @ output, np.eye(5))


def test_matrix_minor():

    A = np.array([[3, 13, 14, 10, 12],
                  [8, 15, 4, 16, 5],
                  [6, 11, 7, 9, 17]])  # <- eliminate this row (idx = 2)
    #                  \
    #                   eliminate this column (idx = 1)

    output = matrix_minor(A, 2, 1)
    expected = np.array([[3, 14, 10, 12],
                         [8, 4, 16, 5]])

    assert np.allclose(output, expected)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
