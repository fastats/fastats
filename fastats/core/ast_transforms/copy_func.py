
from copy import copy
import types
import functools


def copy_func(f, new_funcs):
    """
    Based on http://stackoverflow.com/a/6528148/190597
    by Glenn Maynard

    >>> def f(a, b, c): return a + b + c
    >>> g = copy_func(f, {})
    >>> g is not f
    True
    >>> f(1, 2, 3) == g(1, 2, 3)
    True
    >>> isinstance(np.sin, np.ufunc)
    True
    >>> copy_func(np.sin, {})
    <ufunc 'sin'>
    """
    if not hasattr(f, '__globals__'):
        return f
    globs = copy(f.__globals__)
    globs.update(new_funcs)
    g = types.FunctionType(
        f.__code__, globs, name=f.__name__,
        argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
