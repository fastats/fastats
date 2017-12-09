
import numpy as np

from pytest import approx

from fastats import single_pass
from fastats.maths import relu, softplus


def test_relu_basic_sanity():
    data = np.arange(-2, 3, dtype='float32')

    result = single_pass(data, value=relu)

    assert result[0] == 0.
    assert result[1] == 0.
    assert result[2] == 0.
    assert result[3] == 1.
    assert result[4] == 2.


def test_relu_with_nan_and_inf():
    data = np.array([np.nan, -np.inf, np.inf], dtype='float32')

    result = single_pass(data, value=relu)

    assert np.isnan(result[0])
    assert result[1] == 0.
    assert result[2] == np.inf


def test_softplus_basic_sanity():
    data = np.array([-2, -1, 0, 1, 2], dtype='float32')

    result = single_pass(data, value=softplus)

    assert result[0] == approx(0.12692805)
    assert result[1] == approx(0.31326166)
    assert result[2] == approx(0.69314718)
    assert result[3] == approx(1.31326163)
    assert result[4] == approx(2.12692809)


def test_softplus_with_nan_and_inf():
    data = np.array([np.nan, -np.inf, np.inf], dtype='float32')

    result = single_pass(data, value=softplus)

    assert np.isnan(result[0])
    assert result[1] == 0.
    assert result[2] == np.inf


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
