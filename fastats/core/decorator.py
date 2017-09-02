
from contextlib import contextmanager, suppress
from functools import wraps
from inspect import isfunction

from fastats.core.ast_transforms.convert_to_jit import (
    convert_to_jit
)
from fastats.core.ast_transforms.processor import (
    AstProcessor
)


@contextmanager
def code_transform(func, replaced):
    try:
        yield func
    finally:
        for k, v in replaced.items():
            func.__globals__[k] = v
        replaced.clear()


def fs(func):
    # The initial function *must* be jittable,
    # else we can't do anything.
    _func = func
    replaced = {}

    @wraps(func)
    def fs_wrapper(*args, **kwargs):
        debug = kwargs.get('debug')
        return_callable = kwargs.get('return_callable')

        with suppress(KeyError):
            del kwargs['return_callable']

        if not kwargs:
            return _func(*args)

        with code_transform(_func, replaced) as _f:
            # TODO : remove fastats keywords such as 'debug'
            # before passing into AstProcessor

            new_funcs = {}
            for v in kwargs.values():
                if isfunction(v) and v.__name__ not in kwargs:
                    new_funcs[v.__name__] = convert_to_jit(v)
            kwargs = {k: convert_to_jit(v) for k, v in kwargs.items()}

            processor = AstProcessor(_f, kwargs, replaced, new_funcs)
            proc = processor.process()
            if return_callable:
                return convert_to_jit(proc)
            return convert_to_jit(proc)(*args)

    return fs_wrapper
