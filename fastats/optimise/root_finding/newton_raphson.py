
from fastats.core.decorator import fs
from fastats.maths import deriv


def root(x):
    return x


@fs
def newton_raphson(x0, delta):
    last_x = x0
    next_x = last_x + 10 * delta
    while abs(last_x - next_x) > delta:
        new_y = root(next_x)
        last_x = next_x
        next_x = last_x - new_y / deriv(last_x, delta)
    return next_x


if __name__ == '__main__':
    import pytest
    pytest.main()
