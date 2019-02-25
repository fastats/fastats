
from numpy import power
from fastats.maths import Gamma


def beta_pdf(x, alpha, beta):
	"""
	Beta Distribution Probability
	Density Function.

	PDF across `x` for a beta distribution with parameters
	`alpha` and `beta`.

	It is expected that x be in the range 0<x<1 and
	parameters alpha and beta both be greater than 0.

	>>> beta_pdf(x=0.56, alpha=1.2, beta=3.4)
	0.6068862...
	>>>> beta_pdf(x=0.2, alpha=2, beta=2) # symmetrical
	0.9600000...
	>>> beta_pdf(x=0.8, alpha=2, beta=2) # symmetrical
	0.9600000...
	"""

    u = Gamma(alpha + beta)*power(1 - x, beta - 1)*power(x, alpha - 1)
    v = Gamma(alpha)*Gamma(beta)
    out = u/v
    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
