
from inspect import Signature, Parameter


def build_sig(items: dict) -> Signature:
    """
    Builds a `Signature` object for the parameters
    passed in `items`.

    >>> build_sig({'a': None, 'b': None})
    <Signature (a, b)>
    """
    param = Parameter.POSITIONAL_OR_KEYWORD
    return Signature(Parameter(name, param) for name in items)

