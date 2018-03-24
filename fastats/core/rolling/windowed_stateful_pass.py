
import numpy as np

from fastats.core.decorator import fs


def value(x, val_in, val_out, state):  # pragma: no cover
    return val_in, state


@fs
def windowed_stateful_pass(x, win):
    """
    Performs a *stateful* rolling (windowed)
    iteration over the first dimension of `x`.

    This allows to improve performance by avoiding
    re-calculating values from scratch for each
    window.

    This has the exact same properties as
    :func:`windowed_pass`, but expects the
    ``value`` function be "stateful" and have this
    return signature:
    ``(window_arr, val_in, val_out, state_arr)``

    ``val_in`` and ``val_out`` are the values
    coming "in" to the window and already thrown
    "out" of the window in each iteration.

    The state starts off as an empty array, then
    the first return from the ``value`` function
    sets the state by returning it along with the
    window's value.  The function is free to
    modify the state array in each iteration.

    In the first iteration, you should pre-allocate
    the state array to fit all the state variables
    you will need in subsequent iterations.  Due
    to optimizations made here, only first state
    return will be respected. All subsequent
    iterations can modify the array, but their
    state return value will be ignored.

    Example
    -------
    >>> import numpy as np
    >>> def rolling_sum(x, val_in, val_out, state):
    ...     if state.size == 0:   # state is empty
    ...         state = np.array([np.sum(x)], dtype=x.dtype)
    ...     else:
    ...         state[0] += val_in - val_out
    ...     return state[0], state
    >>> x = np.arange(7, dtype='float')
    >>> result = windowed_stateful_pass(x, 4, value=rolling_sum)
    >>> result
    array([ nan,  nan,  nan,   6.,  10.,  14.,  18.])
    """
    result = np.full_like(x, np.nan)
    state = np.empty(0, dtype=result.dtype)

    val_in = x[win-1]
    val_out = np.nan
    # must call the value func to set the state
    result[win-1], state = value(x[0:win], val_in, val_out, state)

    # subsequent iterations ignore state returns
    # (without this, numba won't optimize out the assignment
    #  and this will be slow)
    for i in range(win+1, x.shape[0]+1):
        start = i - win
        in_idx = i - 1

        val_in = x[in_idx]
        val_out = x[start-1]
        result[in_idx], _ = value(x[start:i], val_in, val_out, state)

    return result


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
