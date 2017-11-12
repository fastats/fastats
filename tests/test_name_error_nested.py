"""
This is a test for a specific edge case:

if the function passed as a kwarg has a nested
function, it doesn't get processed properly.

If we change half(x) / 2 to x / 4, it all works fine.
So basically no nested functions in kwargs.
"""
import numpy as np

from fastats import single_pass


def half(x):
    return x / 2


def quarter(x):
    return half(x) / 2


def test_nested_basic_sanity():
    """
    Some trivially nested functions started
    throwing name errors.
    Just using single_pass(value=half) works
    fine, the value() function gets correctly
    replace with the half() function, however
    single_pass(value=quarter) doesn't work,
    it throws a NameError.
    """
    data = np.array([1, 2, 3], dtype='float32')

    # result_half = single_pass(data, value=half)
    # expected_half = [0.5, 1.0, 1.5]
    # assert np.allclose(expected_half, result_half)

    result = single_pass(data, value=quarter)
    expected = [0.25, 0.5, 0.75]
    assert np.allclose(expected, result)


if __name__ == '__main__':
    import pytest
    pytest.main()
