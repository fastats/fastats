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
    throwing name errors, this test ensures we
    don't suffer from the same problems in the
    future.
    Just using single_pass(value=half) worked
    fine, the value() function was correctly
    replaced with the half() function, however
    single_pass(value=quarter) didn't work,
    it threw a NameError.
    """
    data = np.array([1, 2, 3], dtype='float32')
    py_result = [half(x) for x in data]
    expected_half = [0.5, 1.0, 1.5]
    assert py_result == expected_half

    result_half = single_pass(data, value=half)
    assert np.allclose(expected_half, result_half)

    expected_quarter = [0.25, 0.5, 0.75]
    py_quarter = [quarter(x) for x in data]
    assert py_quarter == expected_quarter

    result = single_pass(data, value=quarter)
    assert np.allclose(expected_quarter, result)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
