
from pytest import approx

from fastats.maths import Gamma

def test_gamma_ints():
	assert Gamma(10) == approx(362879.999)
	assert Gamma(5) == approx(23.9999)
	assert Gamma(19) == approx(6402373705727994.0)

def test_gamma_floats():
	assert Gamma(3.141) == approx(2.28671)
	assert Gamma(8.8129) == approx(27069.4467)
	assert Gamma(12.001) == approx(40014424.15708556)
	assert Gamma(0.007812) == approx(127.4386)
	assert Gamma(2.24) == approx(1.1265) # threshold value


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
