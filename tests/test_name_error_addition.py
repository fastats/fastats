"""
These tests were added as we were seeing
NameErrors occasionally when nested ast-transformed
code was jitted using numba.

The same code without jitting ran fine - the addition
of the jit decorator caused the issues.

Do not change the hierarchy below - it's set up to
test some very specific edge cases:

- Highly nested functions passed as kwargs to a fastats
@fs function. Previously these would error as they
would not jit properly.
- Ensures nested and non-nested kwarg functions all
behave the same.
"""
from fastats import fs


def square(x):
    return x * x


def cube(x):
    return square(x) * x


def quad(x):
    return cube(x) * x


def quint(x):
    return quad(x) * x


@fs
def parent(a):
    b = a + 1
    final = quint(b)
    return final


def double(x):
    return x + x


def triple(x):
    # Change this to return x + x + x and
    # the test passes.
    return double(x) + x


def triple_naive(x):
    return x + x + x


def triple_mul(x):
    return 3 * x


def unit(x):
    return 1


def test_python_functions():
    """
    This is useful to ensure nothing has changed
    """
    assert quint(1) == 1
    assert quint(2) == 32
    assert double(2) == 4
    assert triple(2) == 6
    assert triple_naive(5) == 15
    assert triple_mul(6) == 18
    assert unit(2) == 1
    assert parent(2) == 243


def test_nested_function_valid():
    """
    Getting 1024 instead of 768 when
    using triple().
    """
    trip = parent(3, square=triple)
    assert trip == 768  # 12 * 4**3

    trip_4 = parent(4, square=triple)
    assert trip_4 == 1875


def test_triple_naive():
    trip = parent(3, square=triple_naive)
    assert trip == 768

    trip_4 = parent(4, square=triple_naive)
    assert trip_4 == 1875


def test_triple_mul():
    trip = parent(3, square=triple_mul)
    assert trip == 768

    trip_4 = parent(4, square=triple_mul)
    assert trip_4 == 1875


def test_unit():
    unit_square = parent(3, square=unit)
    assert unit_square == 64

    unit_square_4 = parent(4, square=unit)
    assert unit_square_4 == 125

    unit_square_5 = parent(5, square=unit)
    assert unit_square_5 == 216


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
