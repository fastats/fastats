
import numpy as np
from pytest import approx

from fastats import single_pass, clip


def test_basic_sanity():
    data = np.arange(-10, 10)

    assert data[0] == -10
    assert data[1] == -9
    assert data[-1] == 9

    assert clip(5)(10) == 5
    assert clip(5)(-9) == -5
    assert clip(5)(2) == 2

    result = single_pass(data, value=clip(5))

    assert result[0] == -5
    assert result[1] == -5
    expected = np.array(
        [-5, -5, -5, -5, -5, -5, -4, -3, -2, -1,
         0, 1, 2, 3, 4, 5, 5, 5, 5, 5]
    )
    assert np.allclose(result, expected)


def test_values_outside_range_ignored():
    data = np.arange(-10, 10, dtype='float32')

    result = single_pass(data, value=clip(11.0))

    assert np.allclose(result, data)


def test_clip_value_down_casts():
    data = np.arange(-10, 10)  # ints

    result = single_pass(data, value=clip(5.0))  # float

    assert result[0] == -5
    assert result[1] == -5
    assert result[-2] == 5
    assert result[-1] == 5


def test_clip_value_up_casts():
    data = np.arange(-10.0, 10.0, dtype='float32')

    result = single_pass(data, value=clip(6))  # Int

    assert result[0] == approx(-6.0)
    assert result[1] == approx(-6.0)
    assert result[-2] == approx(6.0)
    assert result[-1] == approx(6.0)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
