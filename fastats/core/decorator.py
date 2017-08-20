
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isfunction

from numba import jit

from fastats.core.ast_transforms.processor import (
    AstProcessor
)


@contextmanager
def code_transform(func, replaced, debug=False):
    try:
        yield func
    finally:
        for k, v in replaced.items():
            func.__globals__[k] = v
        replaced.clear()


def fs(func):
    # The initial function *must* be jittable,
    # else we can't do anything.
    _func = deepcopy(func)
    _jit = jit(nopython=True, nogil=True)
    _jit_func = _jit(_func)
    assert callable(_jit_func)

    replaced = {}

    @wraps(func)
    def fs_wrapper(*args, **kwargs):
        debug = kwargs.get('debug')

        if not kwargs:
            # TODO: should be _jit
            return _func(*args)

        with code_transform(_func, replaced, debug=debug) as _f:
            # TODO : remove fastats keywords such as 'debug'
            # before passing into AstProcessor

            new_funcs = {}
            for v in kwargs.values():
                if isfunction(v) and v.__name__ not in kwargs:
                    new_funcs[v.__name__] = v

            processor = AstProcessor(_f, kwargs, replaced, new_funcs)
            proc = processor.process()
            return proc(*args)

    return fs_wrapper
