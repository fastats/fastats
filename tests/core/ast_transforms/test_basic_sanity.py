
from fastats.core.decorator import fs
from tests import cube
from numba import jit


def child(x):
    return x * x


@fs
def parent(a):
    b = 2 * a
    result = child(b)
    return result


def quad(x):
    return cube(x) * x


def test_child_transform_square_to_cube_execution():
    original = parent(2)
    assert original == 16

    result = parent(2, child=cube)
    assert result == 64

    final = parent(2)
    assert final == 16


def test_child_transform_square_to_quadruple():
    original = parent(2)
    assert original == 16

    result = parent(2, child=quad)
    assert result == 256

    final_two = parent(2)
    assert final_two == 16

    final = parent(3)
    assert final == 36


if __name__ == '__main__':
    import pytest
    pytest.main()
