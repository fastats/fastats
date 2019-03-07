
from pytest import approx

from fastats.maths.beta_pdf import beta_pdf


def test_beta_pdf():
    assert beta_pdf(0.1, 2, 5) == approx(1.968299999999)

    neg = beta_pdf(0.2, 2, 2)
    pos = beta_pdf(0.8, 2, 2)

    assert neg == approx(pos)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
