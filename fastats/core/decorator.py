
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
        # TODO : remove this context manager as it's now not needed?
        pass


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

        # This deliberately mutates the kwargs.
        # We don't want to have a fs-decorated function
        # as a kwarg to another, so we undecorate it first.
        for k, v in kwargs.items():
            if hasattr(v, 'undecorated'):
                kwargs[k] = v.undecorated

        # TODO : ensure jit function returned
        if not kwargs:
            return _func(*args)

        with code_transform(_func, replaced) as _f:
            # TODO : remove fastats keywords such as 'debug'
            # before passing into AstProcessor
            new_funcs = {}
            for v in kwargs.values():
                if isfunction(v) and v.__name__ not in kwargs:
                    inner_replaced = {}
                    with code_transform(v, inner_replaced) as g:
                        processor = AstProcessor(g, kwargs, inner_replaced, new_funcs)
                        proc = processor.process()
                        new_funcs[v.__name__] = convert_to_jit(proc)

            new_kwargs = {}
            for k, v in kwargs.items():
                if new_funcs.get(v.__name__):
                    new_kwargs[k] = new_funcs[v.__name__]
            kwargs.update(new_kwargs)

            processor = AstProcessor(_f, kwargs, replaced, new_funcs)
            proc = processor.process()
            if return_callable:
                return convert_to_jit(proc)

            return convert_to_jit(proc)(*args)

    fs_wrapper.undecorated = _func
    return fs_wrapper
