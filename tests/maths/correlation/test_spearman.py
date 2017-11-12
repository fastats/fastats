
import numpy as np
from pytest import approx

from fastats.maths.correlation import spearman


def test_spearman_basic_sanity():
    """
    Tests the Spearman rank correlation using
    the example on wikipedia:
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    """
    iq = np.array([86, 97, 99, 100, 101, 103, 106, 110, 112, 113])
    tv = np.array([0, 20, 28, 27, 50, 29, 7, 17, 6, 12])

    result = spearman(iq, tv)
    assert result == approx(-0.1757575, abs=1e-7)


def test_spearman_nan_result():
    """
    Confirms that any nan value on the way in
    will result in a nan value returned.
    """
    x = np.array([1, 2, 3, 4], dtype='float')
    y = np.array([2, 3, 4, 3], dtype='float')
    assert spearman(x, y) == approx(0.8)

    x[0] = np.nan
    assert spearman(x, y) == approx(0.4)

    x[0] = 1
    y[0] = np.nan
    assert spearman(x, y) == approx(-0.4)

    y[0] = 2
    assert spearman(x, y) == approx(0.8)


def test_spearman_rgs():
    """
    Tests the Spearman rank correlation using
    the Royal Geographical Society example pdf in
    literature/OASpearmansRankExcelGuidePDF.pdf
    """
    width = np.array([0, 50, 150, 200, 250, 300, 350, 400, 450, 500])
    depth = np.array([0, 10, 28, 42, 59, 51, 73, 85, 104, 96])

    assert spearman(width, depth) == approx(0.9757, abs=1e-4)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
