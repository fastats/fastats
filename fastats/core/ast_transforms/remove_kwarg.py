"""
Copyright (c) 2017, Fastats contributors
Licence: MIT Licence
"""

from inspect import signature, Parameter, Signature

from fastats.core.ast_transforms.build_sig import build_sig


def remove_kwarg(sig, name):
    """
    Removes the `name` keyword argument from the
    `func`, and returns a 2-tuple of new signature
    and the removed key-value pair.

    >>> def cube(x, child='test'):
    ...     return x * x * x
    >>> signature(cube).parameters.keys()
    odict_keys(['x', 'child'])
    >>> sig = signature(cube)
    >>> remove_kwarg(sig, 'child')[0]
    <Signature (x)>
    >>> remove_kwarg(sig, 'child')[1]
    ('child', <Parameter "child='test'">)

    Check Failure modes:

    >>> remove_kwarg(sig, 'bad_arg')
    Traceback (most recent call last):
     ...
    AssertionError: Param 'bad_arg' not found

    >>> remove_kwarg(sig, 'x')
    Traceback (most recent call last):
     ...
    AssertionError: Param 'x' is positional
    """
    assert isinstance(sig, Signature), f"First arg is not a Signature"
    assert isinstance(name, str), f"Second arg is not a string"

    param = sig.parameters.get(name)

    # Parameter *must* be POSITIONAL_OR_KEYWORD,
    # we don't accept tuples of arguments, so
    # VAR_POSITIONAL doesn't apply, and we don't
    # accept `*` or `*args`, so KEYWORD_ONLY
    # also doesn't apply.
    assert param is not None, f"Param '{name}' not found"
    assert param.default is not Parameter.empty, f"Param '{name}' is positional"
    assert param.kind is Parameter.POSITIONAL_OR_KEYWORD

    new_params = sig.parameters.copy()
    value = new_params.pop(name)

    return build_sig(new_params), (name, value)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
