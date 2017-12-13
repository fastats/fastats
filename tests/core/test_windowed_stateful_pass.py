
import numpy as np
from numpy.testing import assert_allclose

from fastats import windowed_stateful_pass


def test_windowed_stateful_pass_respects_value():
    def double(x, val_in, val_out, state):  # pragma: no cover
        return val_in * 2, state

    data = np.array(range(101), dtype='float') ** 3
    ret = windowed_stateful_pass(data, 7, value=double)

    assert np.isnan(ret[0:6]).all()
    assert_allclose(ret[6:], data[6:] * 2)


def test_windowed_stateful_pass_val_out():
    def out_flip(x, val_in, val_out, state):  # pragma: no cover
        return val_out * -1, state

    data = np.array(range(95), dtype='float') ** 3
    ret = windowed_stateful_pass(data, 9, value=out_flip)

    assert np.isnan(ret[0:8]).all()
    assert_allclose(ret[9:], data[:-9] * -1)


def test_windowed_stateful_pass_constant():
    def mean_range(x, val_in, val_out, state):  # pragma: no cover
        state = np.empty(5, dtype=x.dtype)
        state[:] = [1, 2, 3, 4, 5]

        return state.mean(), state

    data = np.array(range(50), dtype='float')
    ret = windowed_stateful_pass(data, 4, value=mean_range)

    assert np.isnan(ret[0:3]).all()
    assert_allclose(ret[3:], 3.0)


def test_windowed_stateful_pass_rolling_mean():
    def rolling_mean(x, val_in, val_out, state):  # pragma: no cover
        if state.size == 0:
            window_sum = np.sum(x)
            state = np.array([window_sum])
        else:
            state[0] += val_in - val_out

        return state[0] / len(x), state

    data = np.array([1, 2, 3, 5, 8, 13], dtype='float')
    ret = windowed_stateful_pass(data, 2, value=rolling_mean)

    assert np.isnan(ret[0])
    assert_allclose(ret[1:], np.array([1.5, 2.5, 4.0, 6.5, 10.5]))


def test_windowed_stateful_pass_callable():
    def triple(x, val_in, val_out, state):  # pragma: no cover
        return val_in * 3, state

    call = windowed_stateful_pass(value=triple, return_callable=True)

    data = np.array([2, 4, 6], dtype='float')
    ret = call(data, 2)

    assert_allclose(ret, [np.nan, 12, 18])


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
