import numpy as np
from nose.tools import assert_true
from scipy import special

from fastats.maths.erf import erf


def test_erf_basic_sanity():
    """
    Taken from literature directory 'error_functions.pdf'
    which is from U. Waterloo, Canada.
    """
    test_data = ((0.0, 0.0000000000),
                 (0.0, 0.0000000000),
                 (0.5, 0.5204998778),
                 (1.0, 0.8427007929),
                 (1.5, 0.9661051465),
                 (2.0, 0.9953222650),
                 (2.5, 0.9995930480),
                 (3.0, 0.9999779095),
                 (3.5, 0.9999992569),
                 (4.0, 0.9999999846),
                 (4.5, 0.9999999998))

    x, expected = zip(*test_data)
    output = erf(np.array(x))
    assert_true(np.allclose(expected, output, atol=1.5e-7))  # max error 1.5e-7


def test_erf_array_input_reconcile_to_scipy():
    x = np.linspace(-5.0, 5.0, 400)
    expected = special.erf(x)
    output = erf(x)
    assert_true(np.allclose(expected, output, atol=1.5e-7))  # max error 1.5e-7


if __name__ == '__main__':
    import nose
    nose.runmodule()