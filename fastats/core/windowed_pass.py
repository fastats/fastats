
import numpy as np

from fastats.core.decorator import fs


def value(x):
    return x[0]


@fs
def windowed_pass(x, win):
    """
    Performs a rolling (windowed) iteration
    over the first dimension of `x`.

    This is a simplistic rolling window which
    does not take into account NaN values,
    and maintains the same fixed size window
    over all data points.

    The leading values will be nan up until
    the window size.

    Example
    -------

    >>> import numpy as np
    >>> def mean(x):
    ...     return np.sum(x) / x.size
    >>> x = np.array(range(10), dtype='float')
    >>> result = windowed_pass(x, 5, value=mean)
    >>> result[:6]
    array([nan, nan, nan, nan, 2., 3.])
    >>> result[6:10]
    array([4., 5., 6., 7.])
    """
    result = np.full_like(x, np.nan)
    for i in range(win, x.shape[0]+1):
        result[i-1] = value(x[i-win:i])
    return result


@fs
def windowed_pass_2d(x, win):
    """
    The same as windowed pass, but explicitly
    iterates over the `value()` return array
    and allocates it in the `result`.

    This allows 2-dimensional arrays to be returned
    from `value()` functions, before we support
    the behaviour properly using AST transforms.

    This allows for extremely fast iteration
    for items such as OLS, and at the same time
    calculating t-stats / r^2.
    """
    result = np.full_like(x, np.nan)
    for i in range(win, x.shape[0]+1):
        res = value(x[i-win:i])
        for j, j_val in enumerate(res):
            result[i-1, j] = j_val
    return result


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
