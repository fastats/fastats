
from fastats.core.decorator import fs


def test_decorated_func_kwargs():
    @fs
    def square(x):
        return x ** 2

    @fs
    def func_that_takes_func_kwarg(arg, func=abs):
        return func(arg)

    func_that_takes_func_kwarg(4, func=square)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
