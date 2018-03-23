
import pytest
from fastats.core.decorator import fs


@pytest.mark.xfail
def test_decorated_func_kwargs():
    @fs
    def square(x):
        return x ** 2

    @fs
    def func_that_takes_func_kwarg(arg, func=abs):
        return func(arg)

    result = func_that_takes_func_kwarg(4, func=square)
    assert result == 16


def test_no_decorator_func_kwargs():
    def square(x):
        return x ** 2

    def func_that_takes_func_kwarg(arg, func=abs):
        return func(arg)

    result = func_that_takes_func_kwarg(4, func=square)
    assert result == 16

if __name__ == '__main__':
    pytest.main([__file__])
