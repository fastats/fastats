
import numpy as np
from numpy.testing import assert_allclose

from fastats import windowed_stateful_pass


def test_windowed_stateful_pass_raw():
    data = np.array(range(11), dtype='float') ** 3

    # not using jit, falling back to "value" function
    # defined in windowed_stateful_pass module
    raw = windowed_stateful_pass(data, 3)

    assert np.isnan(raw[0:2]).all()
    assert_allclose(raw[2:], data[2:])


def double(x, val_in, val_out, state):
    return val_in * 2, state


def test_windowed_stateful_pass_respects_value():
    data = np.array(range(101), dtype='float') ** 3
    assert double(data, 1, 2, data) == (2, data)

    ret = windowed_stateful_pass(data, 7, value=double)

    assert np.isnan(ret[0:6]).all()
    assert_allclose(ret[6:], data[6:] * 2)


def out_flip(x, val_in, val_out, state):
    return val_out * -1, state


def test_windowed_stateful_pass_val_out():
    data = np.array(range(95), dtype='float') ** 3
    assert out_flip(data, 1, 1, data) == (-1, data)

    ret = windowed_stateful_pass(data, 9, value=out_flip)

    assert np.isnan(ret[0:8]).all()
    assert_allclose(ret[9:], data[:-9] * -1)


def mean_range(x, val_in, val_out, state):
    state = np.empty(5, dtype=x.dtype)
    state[:] = [1, 2, 3, 4, 5]

    return state.mean(), state


def test_windowed_stateful_pass_constant():
    data = np.array(range(50), dtype='float')
    mean_result = mean_range(data, 66, 66, data)

    assert mean_result[0] == 3.0
    assert_allclose(mean_result[1], np.array([1, 2, 3, 4, 5], dtype='float'))

    ret = windowed_stateful_pass(data, 4, value=mean_range)

    assert np.isnan(ret[0:3]).all()
    assert_allclose(ret[3:], 3.0)


def rolling_mean(x, val_in, val_out, state):
    if state.size == 0:
        window_sum = np.sum(x)
        state = np.array([window_sum])
    else:
        state[0] += val_in - val_out

    return state[0] / len(x), state


def test_windowed_stateful_pass_rolling_mean():
    data = np.array([1, 2, 3, 5, 8, 13], dtype='float')

    # testing rolling_mean state handling
    empty_state = np.empty(0, dtype=data.dtype)

    mean, state = rolling_mean(data[:-1], 0, 0, empty_state)
    expected_state = np.array(data[:-1].sum())
    assert mean, state == (data[:-1].mean(), expected_state)

    mean2, state2 = rolling_mean(data[1:], 13, 1, state)
    expected_state_2 = np.array(data[1:].sum())
    assert mean2, state2 == (data[1:].mean(), expected_state_2)

    ret = windowed_stateful_pass(data, 2, value=rolling_mean)

    assert np.isnan(ret[0])
    assert_allclose(ret[1:], np.array([1.5, 2.5, 4.0, 6.5, 10.5]))


def triple(x, val_in, val_out, state):
    return val_in * 3, state


def test_windowed_stateful_pass_callable():
    data = np.array([2, 4, 6], dtype='float')
    assert triple(data, 3, 3, data) == (9, data)

    call = windowed_stateful_pass(value=triple, return_callable=True)
    ret = call(data, 2)

    assert_allclose(ret, [np.nan, 12, 18])


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
