
import nose
from numpy.testing import assert_almost_equal

from fastats.maths.norm_cdf import norm_cdf


def test_norm_cdf_basic_sanity():
    assert_almost_equal(0.5, norm_cdf(0.0, 0, 1))


def test_norm_cdf_dartmouth():
    """
    Examples taken from:
    https://math.dartmouth.edu/archive/m20f12/public_html/matlabnormal
    stored in literature directory as dartmouth_normcdf_norminv.pdf
    """
    assert_almost_equal(0.0062, norm_cdf(90, 100, 4), decimal=4)


if __name__ == '__main__':
    nose.runmodule()