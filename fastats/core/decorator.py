
from contextlib import contextmanager, suppress
from functools import wraps
from inspect import isfunction, isbuiltin

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
        return_callable = kwargs.pop('return_callable', None)

        if not kwargs:
            return _func(*args)

        with code_transform(_func, replaced) as _f:
            # TODO : remove fastats keywords such as 'debug'
            # before passing into AstProcessor

            new_funcs = {}
            for v in kwargs.values():
                if v.__name__ in kwargs:
                    continue

                if isfunction(v):
                    new_funcs[v.__name__] = convert_to_jit(v)
                elif isbuiltin(v):
                    new_funcs[v.__name__] = v

            kwargs = {k: convert_to_jit(v) if not isbuiltin(v) else v
                      for k, v in kwargs.items()}

            processor = AstProcessor(_f, kwargs, replaced, new_funcs)
            proc = processor.process()
            if return_callable:
                return convert_to_jit(proc)
            return convert_to_jit(proc)(*args)

    return fs_wrapper
