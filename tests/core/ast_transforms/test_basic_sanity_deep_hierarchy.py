
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
    print(f'Double called with {x}')
    return x + x


def triple(x):
    return double(x) + x


# def test_top_level_square_to_add():
#     original = fs(cube)(3)
#     assert original == 3**3
#
#     new = fs(cube)(3, square=double)
#     assert new == 6 * 3
#
#     # Ensure the standalone functions
#     # still work as normal.
#     assert square(3) == 9
#     assert cube(3) == 27


# def test_second_level_square_to_add():
#     # original = fs(quad)(3)
#     # assert original == 3**4
#
#     # TODO : this fails.
#     # potentially doesn't like
#     # nested hierarchies.
#     new = fs(quad)(3, square=double)
#     assert new == 6 * 3**2

#
# def test_square_to_add_ast_replace():
#     """
#     This confirms that the nested AST
#     replacement actually works.
#     """
#     add = parent(2, square=double)
#     assert add == 2 * 3**4


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


# def test_multiple_ast_replacements():
#     # Does the normal function work?
#     original = parent(3)
#     assert original == 4**5
#
#     add = parent(3, square=double)
#     assert add == 2 * 4**4
#
#     triple = parent(3, square=triple)
#     assert add == ???
#
#     final = parent(2)
#     assert final == 3**5


if __name__ == '__main__':
    import pytest
    pytest.main()
