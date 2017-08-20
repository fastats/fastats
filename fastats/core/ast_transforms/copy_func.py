import types
import functools


def copy_func(f):
    """
    Based on http://stackoverflow.com/a/6528148/190597
    by Glenn Maynard

    >>> def f(a, b, c): return a + b + c
    >>> g = copy_func(f)
    >>> g is not f
    True
    >>> f(1, 2, 3) == g(1, 2, 3)
    True
    """
    g = types.FunctionType(
        f.__code__, f.__globals__, name=f.__name__,
        argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g