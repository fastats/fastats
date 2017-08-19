
from inspect import Signature, Parameter, signature

import pytest

from fastats.core.ast_transforms.remove_kwarg import remove_kwarg


def cube_one_arg(x, child='test'):
    return x * x * x


def cube_two_arg(x, y, child='test'):
    return x * x * x


def cube_two_two(x, y, first='test', second='test2'):
    return x * x * x


def test_one_arg_removal():
    sig = signature(cube_one_arg)
    new_sig, removed = remove_kwarg(sig, 'child')

    assert isinstance(new_sig, Signature)
    assert len(new_sig.parameters) == 1
    assert isinstance(removed, tuple)
    assert len(removed) == 2
    assert removed[0] == 'child'

    kind = Parameter.POSITIONAL_OR_KEYWORD
    expect = Parameter('child', kind=kind, default='test')
    assert removed[1] == expect


def test_one_arg_fails_on_nonexistent_kwarg():
    with pytest.raises(AssertionError) as excinfo:
        sig = signature(cube_one_arg)
        _, _ = remove_kwarg(sig, 'bad_arg')

    excinfo.match(r"Param 'bad_arg' not found")


def test_one_arg_fails_on_positional_removal():
    with pytest.raises(AssertionError) as excinfo:
        sig = signature(cube_one_arg)
        _, _ = remove_kwarg(sig, 'x')

    excinfo.match(r"Param 'x' is positional")


def test_one_arg_fails_if_arguments_switched():
    with pytest.raises(AssertionError) as excinfo:
        sig = signature(cube_one_arg)
        _, _ = remove_kwarg('child', sig)

    excinfo.match(r"First arg is not a Signature")


def test_two_arg_removal():
    sig = signature(cube_two_arg)
    new_sig, removed = remove_kwarg(sig, 'child')

    assert isinstance(new_sig, Signature)
    assert len(new_sig.parameters) == 2
    assert set(new_sig.parameters) == {'x', 'y'}


def test_two_arg_fails_on_nonexistent_kwarg():
    with pytest.raises(AssertionError) as excinfo:
        sig = signature(cube_two_arg)
        _, _ = remove_kwarg(sig, 'also_bad')

    excinfo.match(r"Param 'also_bad' not found")


def test_two_arg_fails_on_positional_removal():
    with pytest.raises(AssertionError) as excinfo:
        sig = signature(cube_two_arg)
        _, _ = remove_kwarg(sig, 'y')

    excinfo.match(r"Param 'y' is positional")


def test_kwarg_removal_leaves_other_kwargs():
    sig = signature(cube_two_two)
    new_sig, removed = remove_kwarg(sig, 'first')

    assert isinstance(new_sig, Signature)
    assert len(new_sig.parameters) == 3
    assert set(new_sig.parameters) == {'x', 'y', 'second'}


if __name__ == '__main__':
    import pytest
    pytest.main()