
import pytest

from fastats.core.decorator import fs
from tests import cube


def child(x):
    return x * x


@fs
def parent(a):
    b = 2 * a
    result = child(b)
    return result


def quad(x):
    return cube(x) * x


def zero(x):
    return 0


def child_faker(x):
    return 42


child_faker.__name__ = 'child'


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


def test_child_transform_square_to_zero():
    original = parent(2)
    assert original == 16

    assert zero("ignored") == 0

    result = parent(2, child=zero)
    assert result == 0

    final_two = parent(2)
    assert final_two == 16

    final = parent(3)
    assert final == 36


def test_problematic_child_transform_with_faked_child():
    # function with a faked name should be respected as an override
    # TODO this test captures current problematic behaviour which should be fixed
    assert child_faker.__name__ == child.__name__

    original = parent(1)
    assert original == 4

    assert child_faker("ignored") == 42

    result = parent(1, child=child_faker)
    if result == 4:
        pytest.xfail("Known problem (function with faked name not respected as override)")

    # TODO remove the pragma once test is passing (masking unreachable code)
    else:   # pragma: no cover
        assert result == 42

        final = parent(1)
        assert final == 4

        pytest.fail("Unexpectedly passed (did you fix the code and forgot to update this test?)")


if __name__ == '__main__':
    import pytest
    pytest.main()
