
from fastats.core.ast_transforms.processor import AstProcessor


def single(x):
    return x


def double(x):
    return single(x) + single(x)


def triple(x):
    return double(x) + single(x)


def twice(x):
    return x + x


def test_overrides():
    assert triple(1) == 3
    assert triple(2) == 6

    replaced = {}
    changes = {'single': twice}
    proc = AstProcessor(triple, changes, replaced)
    new_func = proc.process()

    assert new_func(1) == 6
    assert 'double' in replaced
    assert 'single' in replaced

    # If the assert below fails, then the actual
    # triple() function has been modified in the
    # original globals() dict, instead of in a copy.
    assert triple(1) == 3

    # Once the replaced items go back
    # into globals, we must get normal behaviour
    for k, v in replaced.items():
        triple.__globals__[k] = v

    assert triple(1) == 3
    assert triple(2) == 6


def test_without_overrides():
    assert triple(1) == 3
    assert triple(2) == 6

    replaced = {}
    proc = AstProcessor(triple, {}, replaced)
    new_func = proc.process()

    assert new_func(1) == 3
    assert new_func(2) == 6
    assert 'single' in replaced
    assert 'double' in replaced
    assert triple(1) == 3
    assert double(1) == 2
    assert single(1) == 1


if __name__ == '__main__':
    import pytest
    pytest.main()
