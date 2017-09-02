
from pytest import approx

from fastats.maths import erf


def test_erf_basic_sanity():
    """
    Taken from literature directory 'error_functions.pdf'
    which is from U. Waterloo, Canada.
    """
    assert erf(0.0) == approx(0.0, abs=1e-9)
    assert erf(0.5) == approx(0.5204998778)
    assert erf(1.0) == approx(0.8427007929)
    assert erf(1.5) == approx(0.9661051465)
    assert erf(2.0) == approx(0.9953222650)
    assert erf(2.5) == approx(0.9995930480)
    assert erf(3.0) == approx(0.9999779095)
    assert erf(3.5) == approx(0.9999992569)
    assert erf(4.0) == approx(0.9999999846)
    assert erf(4.5) == approx(0.9999999998)


if __name__ == '__main__':
    import pytest
    pytest.main()
