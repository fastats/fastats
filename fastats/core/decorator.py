
from collections import defaultdict
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import NamedTuple, Any

from numba import jit

from fastats.core.ast_transforms.processor import (
    AstProcessor
)


@contextmanager
def code_transform(func, replaced, debug=True):
    original = deepcopy(func)
    try:
        yield original
    finally:
        # import pdb; pdb.set_trace()
        for k, v in replaced.items():
            print(f'Replacing: {k} in globals')
            func.__globals__[k] = v



def fs(func):
    # The initial function *must* be jittable,
    # else we can't do anything.
    _func = deepcopy(func)
    _original_code = _func.__code__

    # _func.__code__ = _original_code
    # _jit = jit(nopython=True, nogil=True)
    # _jit_func = _jit(_func)
    # assert callable(_jit_func)

    replaced = {}

    def fs_wrapper(*args, **kwargs):
        # TODO: why is this necessary?
        _func.__code__ = _original_code
        debug = kwargs.get('debug')

        with code_transform(_func, replaced, debug=debug) as _f:
            if not kwargs:
                # TODO: should be _jit_func
                return _f(*args)

            processor = AstProcessor(_f, kwargs, replaced)
            proc = processor.process()
            return proc(*args)

    return fs_wrapper
