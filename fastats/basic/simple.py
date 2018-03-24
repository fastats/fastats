
from fastats import fastfunc


__all__ = (
    "zero",
    "one",
    "double",
    "triple",
    "square",
    "cube",
    "identity",
    "flip",
    "invert",
)


# constants


@fastfunc
def zero():
    return 0


@fastfunc
def one():
    return 1


# multiplication


@fastfunc
def double(x):
    return 2 * x


@fastfunc
def triple(x):
    return 3 * x


# exponentiation


@fastfunc
def square(x):
    return x ** 2


@fastfunc
def cube(x):
    return x ** 3


# basic value manipulation


@fastfunc
def identity(x):
    return x


@fastfunc
def flip(x):
    return -1 * x


@fastfunc
def invert(x):
    assert x != 0

    return 1 / x


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
