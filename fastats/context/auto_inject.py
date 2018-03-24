
import contextlib


@contextlib.contextmanager
def auto_inject(recursive: bool = True):
    """
    Context which makes it possible to
    automatically "inject" objects into fastfunc
    calls by passing in extra named arguments.

    .. note::
        This essentially changes Python semantics
        when active.  We inherently break
        Python call semantics and enable "ad-hoc"
        substitutions, without spelling out
        all the substitutions to perform.

    Examples:

    >>> TO DO
    >>> EXAMPLES
    """
    yield
