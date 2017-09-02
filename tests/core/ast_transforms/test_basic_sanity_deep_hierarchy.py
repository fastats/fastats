
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
    # TODO : investigate double(x) + x failure.
    return 3*x


def unit(x):
    return 1


def test_top_level_square_to_add():
    original = fs(cube)(3)
    assert original == 3**3

    new = fs(cube)(3, square=double)
    assert new == 6 * 3

    # Ensure the standalone functions
    # still work as normal.
    assert square(3) == 9
    assert cube(3) == 27


def test_second_level_square_to_add():
    original = fs(quad)(3)
    assert original == 3**4

    new = fs(quad)(3, square=double)
    assert new == 6 * 3**2

    assert quad(3) == 81
    assert cube(3) == 27


def test_square_to_add_ast_replace():
    """
    This confirms that the nested AST
    replacement actually works.
    """
    add = parent(2, square=double)
    assert add == 2 * 3**4

    assert double(2) == 4


def test_square_to_add_no_side_effects():
    """
    This tests that after a successful AST
    replacement, the original function still
    performs as expected, without the
    modifications.
    """
    # Does the normal function work?
    original = parent(2)
    assert original == 3**5

    # Does square() get changed?
    add = parent(2, square=double)
    assert add == 2 * 3**4

    # Is the function normal again?
    final = parent(3)
    assert final == 4**5


def test_multiple_ast_replacements():
    """
    `parent` function takes `n`, adds 1
    and then takes that to the fifth power.
    """
    # Does the normal function work?
    original = parent(3)
    assert original == 1024  # (3 + 1)**5

    add = parent(3, square=double)
    assert add == 512  # 2 * (3 + 1)**4

    single = parent(3, square=unit)
    assert single == 64  # 1 * (3 + 1)**3

    trip = parent(3, square=triple)
    assert trip == 768  # 12 * 4**3

    orig_3 = parent(3)
    assert orig_3 == 1024  # (3 + 1)**5

    orig_4 = parent(4)
    assert orig_4 == 3125  # (4 + 1)**5

    final = parent(2)
    assert final == 243  # (2 + 1)**5

    assert unit(3) == 1
    assert double(3) == 6
    assert triple(3) == 9
    assert square(4) == 16


if __name__ == '__main__':
    import pytest
    pytest.main()
