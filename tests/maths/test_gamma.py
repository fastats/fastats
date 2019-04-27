
from pytest import approx, raises

from fastats.maths.gamma import gammaln


def test_gamma_ints():
    assert gammaln(10) == approx(12.801827480081469, rel=1e-6)
    assert gammaln(5) == approx(3.1780538303479458, rel=1e-6)
    assert gammaln(19) == approx(36.39544520803305, rel=1e-6)


def test_gamma_floats():
    assert gammaln(3.141) == approx(0.8271155090776673, rel=1e-6)
    assert gammaln(8.8129) == approx(10.206160943471318, rel=1e-6)
    assert gammaln(12.001) == approx(17.50475055100354, rel=1e-6)
    assert gammaln(0.007812) == approx(4.847635060148693, rel=1e-6)
    assert gammaln(86.13) == approx(296.3450079998172, rel=1e-6)


def test_gamma_negative():
    raises(AssertionError, gammaln, -1)
    raises(AssertionError, gammaln, -0.023)
    raises(AssertionError, gammaln, -10.9)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
