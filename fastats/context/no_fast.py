
import contextlib


@contextlib.contextmanager
def no_fast():
    yield
