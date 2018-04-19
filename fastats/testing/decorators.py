
import inspect

import pytest

from fastats.context.no_fast import no_fast


def fast_no_fast(func):
    signature = inspect.signature(func)
    takes_is_fast = "is_fast" in signature.parameters

    # We deliberately avoid functools.wraps()
    # because pytest collection might fail here
    # if the wrapped function (pretending to be
    # func) does not take "is_fast" parameter.
    #
    # This is because functools.wraps replaces
    # the signature of the wrapped function
    # to looks like the original function.
    def wrapped(is_fast, *args, **kwargs):
        extra_kwargs = {"is_fast": is_fast} if takes_is_fast else {}

        if is_fast:
            return func(*args, **kwargs, **extra_kwargs)
        else:
            with no_fast():
                return func(*args, **kwargs, **extra_kwargs)

    parametrizer = pytest.mark.parametrize(
        "is_fast", (True, False), ids=("fast", "no_fast")
    )

    return parametrizer(wrapped)
