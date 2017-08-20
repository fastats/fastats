
import nose
from pytest import approx

from fastats.maths.erfc import erfc


def test_erfc_basic_sanity():
    assert erfc(0.0) == approx(1.0)


def test_erfc_matlab_examples():
    """
    Examples taken from:
    https://uk.mathworks.com/help/matlab/ref/erfc.html
    """
    assert erfc(0.35) == approx(0.6206179)


def test_erfc_ibm_examples():
    """
    Examples taken from:
    http://www.ibm.com/support/knowledgecenter/SSLTBW_2.1.0/com.ibm.zos.v2r1.bpxbd00/erf.htm
    """
    assert erfc(10.0) == approx(2.0884877e-46)
    assert erfc(0.1) == approx(0.88753707)


def test_erfc_negative_input():
    assert erfc(-1.0) == approx(1.84270078)


if __name__ == '__main__':
    nose.runmodule()