
from functools import wraps
from inspect import isfunction

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit
from fastats.core.ast_transforms.processor import AstProcessor


def fs(func):
    """
    This is the decorator which performs recursive AST substitution of
    functions, and optional JIT-compilation using `numba`_.

    This must only be used on functions with positional parameters
    defined; this must not be used on functions with keyword parameters.

    This decorator modifies the original function (and any nested
    function calls) by replacing any functions passed in using keyword
    arguments. It replaces them in the AST and returns a new function
    object with a new code object that calls the replacement functions
    instead.

    For example, a function hierarchy such as:

    >>> def calculate(x):
    ...     return x * x
    >>> def my_func(x):
    ...     a = calculate(x)
    ...     return a / 2

    will take the input variable `x`, square it, and then halve the
    result:

    >>> my_func(6)
    18.0

    Six squared is 36, divided by two is 18.

    If you wanted to replace the `calculate` function to
    return a different calculation, you could use this `@fs`
    decorator:

    >>> @fs
    ... def my_func(x):
    ...     a = calculate(x)
    ...     return a / 2

    Now the `my_func` callable is able to accept keyword arguments,
    which it will replace recursively throughout its hierarchy.

    If you wanted to change the `calculate` in this function to:

    >>> def cube(x):
    ...     return x * x * x

    then after applying the `@fs` decorator you can do this:

    >>> my_func(6, calculate=cube)
    108.0

    Six cubed is 216, divided by two is 108.0.

    This parametrisation can be decided at runtime - every time a
    new keyword argument is passed in, it generates a new function
    object with a new code object.

    To store the new function object instead of executing it, pass
    `return_callable=True` to the decorated function:

    >>> new_func = my_func(6, calculate=cube, return_callable=True)
    >>> # At this point the new function has **not** been called.
    >>> new_func(6)
    108.0
    """
    _func = func
    replaced = {}

    @wraps(func)
    def fs_wrapper(*args, **kwargs):
        return_callable = kwargs.pop('return_callable', None)

        # This deliberately mutates the kwargs.
        # We don't want to have a fs-decorated function
        # as a kwarg to another, so we undecorate it first.
        for k, v in kwargs.items():
            if hasattr(v, 'undecorated'):
                kwargs[k] = v.undecorated

        # TODO : ensure jit function returned
        if not kwargs:
            return _func(*args)

        # TODO : remove fastats keywords such as 'debug'
        # before passing into AstProcessor
        new_funcs = {}
        for v in kwargs.values():
            if isfunction(v) and v.__name__ not in kwargs:
                inner_replaced = {}
                processor = AstProcessor(v, kwargs, inner_replaced, new_funcs)
                proc = processor.process()
                new_funcs[v.__name__] = convert_to_jit(proc)

        new_kwargs = {}
        for k, v in kwargs.items():
            if new_funcs.get(v.__name__):
                new_kwargs[k] = new_funcs[v.__name__]
        kwargs.update(new_kwargs)

        processor = AstProcessor(_func, kwargs, replaced, new_funcs)
        proc = processor.process()
        if return_callable:
            return convert_to_jit(proc)

        return convert_to_jit(proc)(*args)

    fs_wrapper.undecorated = _func
    return fs_wrapper


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
