
from pytest import approx

from fastats.maths.t_pdf import t_pdf


def test_t_pdf():
    assert t_pdf(0, 3) == approx(0.36755259694)

    neg = t_pdf(-10, 3.14)
    pos = t_pdf(10, 3.14)

    assert neg == approx(pos)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
