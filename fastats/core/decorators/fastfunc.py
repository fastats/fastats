
import functools


def fastfunc(func, **bind_funcs):
    """
    The core "fast" function decorator.

    This serves two purposes:

    1. Produce "fast" function objects from pure
       ones.
    2. Produce compound "fast" functions
       (enabling passing other fastfuncs into
        first one).

    Under the hood, does TODO TODO ...
    """
    if not bind_funcs:
        return func
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs, **bind_funcs)

        return wrapper
