
from fastats.core.decorator import fs


def square(x):
    return x * x


@fs
def cube(x):
    return x * x * x


@fs
def func(x):
    a = square(x)
    return a / 2


def test_fs_decorated_functions_as_kwargs_to_another():
    assert square(2) == 4.0
    assert square(3) == 9.0
    assert cube(3) == 27.0
    assert cube(4) == 64.0

    assert func(6) == 18.0
    assert func(4) == 8.0

    assert func(6, square=cube) == 108.0
    assert func(4, square=cube) == 32.0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
