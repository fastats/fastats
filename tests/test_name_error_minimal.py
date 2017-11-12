"""
This is a minimal example which recreates
the NameError/numba.TypingError seen when
jitting the AST transforms using numba.

The code runs fine without any jit code,
but fails when jitted.
"""

import numpy as np

from fastats import single_pass


def unit(x):
    return x


def twice(x):
    return unit(x) * 2


def test_transform_func_calls_second_func():
    # Default iteration with no transforms
    data = np.array([1, 2, 3])
    py_unit = [unit(x) for x in data]
    assert py_unit == data.tolist()

    default = single_pass(data)

    # The default behaviour should not
    # change the values in the data array
    expected = np.array([1, 2, 3])
    assert np.allclose(expected, default)

    py_twice = [twice(x) for x in data]
    assert py_twice == (data * 2).tolist()

    # Now perform an iteration but change
    # the function for one that doubles.
    result = single_pass(data, value=twice)

    # Replacing the default function for one
    # that doubles the values.
    expected = np.array([2, 4, 6])
    assert np.allclose(expected, result)


if __name__ == '__main__':
    import pytest
    pytest.main()
