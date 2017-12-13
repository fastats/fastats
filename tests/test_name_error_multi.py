
import numpy as np

from fastats import single_pass


def first(x):
    return x * x


def second(x):
    return x * x * x


def multi(x):
    a = first(x)
    b = second(a)
    return b


def test_multiple_transforms_top_level():
    """
    This tests for the 'free vars' issue -
    we had failures when the value= kwarg
    was decorated with @fs, but if you remove
    the decorator it's fine.
    """
    data = np.array([1, 2, 3])
    py_result = [multi(x) for x in data]
    assert py_result == [1, 64, 729]

    result = single_pass(data, value=multi)

    expected = np.array([1, 64, 729])
    assert np.allclose(expected, result)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
