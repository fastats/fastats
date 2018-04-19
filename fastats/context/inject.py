
import contextlib

from typing import Callable, Mapping


@contextlib.contextmanager
def inject(recursive: bool = True,
           force_fastfuncs: bool = True,
           **inject_symbols: Mapping[str, Callable]):
    """
    Context which explicitly injects objects
    into function calls.

    :param recursive: apply injection recursively.
    :param force_fastfuncs: ensure that every object
        injected is a [GLOSSARY REFERENCE: fastfunc]
    :param inject_symbols: mapping of names to
        functions to inject.
    """
    yield
