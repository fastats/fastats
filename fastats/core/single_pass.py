
import numpy as np

from fastats.core.decorator import fs


def value(x):
    return x


@fs
def single_pass(x):
    """
    Performs a single iteration over the first
    dimension of `x`.

    Tests
    -----
    >>> def square(x):
    ...     return x * x
    >>> data = np.arange(10)
    >>> single_pass(data, value=square)
    array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

    >>> import math
    >>> def calc(x):
    ...     return 2 * math.log(x)
    >>> single_pass(data[1:], value=calc)
    array([0, 1, 2, 2, 3, 3, 3, 4, 4])
    """
    result = np.zeros_like(x)
    for i in range(x.shape[0]):
        result[i] = value(x[i])
    return result


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
