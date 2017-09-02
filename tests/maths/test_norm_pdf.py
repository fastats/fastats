
from pytest import approx

from fastats.maths import norm_pdf


def test_norm_pdf_basic_sanity():
    assert norm_pdf(0, 0.0, 1.0) == approx(0.3989422804)

    neg = norm_pdf(-0.1, 0.0, 1.0)
    pos = norm_pdf(0.1, 0.0, 1.0)
    assert neg == approx(pos)


if __name__ == '__main__':
    import pytest
    pytest.main()
