
from functools import wraps
from inspect import isfunction

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit
from fastats.core.ast_transforms.processor import AstProcessor


def fs(func):
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
