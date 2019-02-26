
from pytest import approx, raises

from fastats.maths.gamma import gamma


def test_gamma_ints():
    assert gamma(10) == approx(362879.99999970134, rel=1e-6)
    assert gamma(5) == approx(23.999999990491887, rel=1e-6)
    assert gamma(19) == approx(6402373705727994.0, rel=1e-6)


def test_gamma_floats():
    assert gamma(3.141) == approx(2.286713161252258, rel=1e-6)
    assert gamma(8.8129) == approx(27069.44671814589, rel=1e-6)
    assert gamma(12.001) == approx(40014424.15708556, rel=1e-6)
    assert gamma(0.007812) == approx(127.43864844822373, rel=1e-6)
    assert gamma(2.24) == approx(1.1265656357270413, rel=1e-6)  # threshold value


def test_gamma_negative():
    raises(AssertionError, gamma, -1)
    raises(AssertionError, gamma, -0.023)
    raises(AssertionError, gamma, -10.9)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
