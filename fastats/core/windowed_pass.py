
import numpy as np

from fastats.core.decorator import fs


def value(x):
    return x


@fs
def windowed_pass(x, win):
    """
    Performs a rolling (windowed) iteration
    over the first dimension of `x`.

    This is a simplistic rolling window which
    does not take into account NaN values,
    and maintains the same fixed size over
    all data points.

    Example
    -------

    >>> import numpy as np
    >>> def mean(x):
    ...     return np.sum(x) / x.size
    >>> x = np.array(range(10), dtype='float')
    >>> result = windowed_pass(x, 5, value=mean)
    >>> result[:6]
    array([ nan,  nan,  nan,  nan,   2.,   3.])
    >>> result[6:10]
    array([ 4.,  5.,  6.,  7.])

    :param x:
    :param win:
    :return:
    """
    result = np.full_like(x, np.nan)
    for i in range(win, x.shape[0]+1):
        result[i-1] = value(x[i-win:i])
    return result


if __name__ == '__main__':
    import pytest
    pytest.main()
