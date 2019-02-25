
from numpy import power, sqrt
from fastats.maths import Gamma

def t_pdf(x, nu):
	"""
	Student-T Distribution Probability
	Density Function.

	PDF across `x` for a beta distribution with degrees of
	freedom `nu`.

	It is expected that x be a real number, and that nu
	be a positive real number.

	>>> t_pdf(x=0.56, nu=5)
	0.3796067...
	>>>> t_pdf(x=-5, nu=2) # symmetrical
	0.0071277...
	>>> t_pdf(x=5, nu=2) # symmetrical
	0.0071277...
	"""

    u = Gamma(0.5*(nu + 1))
    v1 = sqrt(nu*pi)*Gamma(0.5*nu)
    v2 = power(1 + power(x,2)/nu, 0.5*(nu + 1))
    out = u/(v1*v2)
    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
