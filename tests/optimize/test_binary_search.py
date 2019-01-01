
from hypothesis import given
from hypothesis.strategies import floats

from numpy import cos, tan
from pytest import approx

from fastats.optimise.binary_search import binary_search
from fastats.optimise.binary_search import root


def func(x):
    return x ** 3 - x - 1


bs_func = binary_search(0.1, 1.9, 1e-6, root=func,
                        return_callable=True)


def test_basic_sanity():
    assert func(0.5) == approx(-1.375)
    no_kwargs = binary_search(-2.0, 2.0, 1e-6)
    assert no_kwargs == approx(0.0)

    no_kwargs2 = binary_search(-1.0, 2.0, 1e-6)
    assert no_kwargs2 == approx(0.0, abs=1e-5)

    no_kwargs3 = binary_search(-0.1, 0.1, 1e-3)
    assert no_kwargs3 == approx(0.0)

    no_kwargs4 = binary_search(-0.1, 10.0, 1e-3)
    assert no_kwargs4 == approx(0.00048217, abs=1e-3)

    value = bs_func(-2.0, 2.0, 1e-6)
    assert value == approx(1.324717)

    res2 = bs_func(-1.0, 2.0, 1e-6)
    assert res2 == approx(1.324717)

    res3 = bs_func(-0.1, 2.0, 1e-6)
    assert res3 == approx(1.324717)

    low_delta = bs_func(-2.0, 2.0, 1e-4)
    assert low_delta == approx(1.324, abs=1e-3)


@given(floats(allow_nan=False))
def test_default_root(n):
    assert root(n) == approx(n)


def test_with_local_function():
    """
    Example taken from literature/NewtonRaphson.pdf
    and the corresponding Newton-Raphson tests.
    """
    def cos_func(x):
        return cos(x) - 2 * x

    assert cos_func(0.5) == approx(-0.12241743)

    value = binary_search(-1.0, 1.0, 1e-6, root=cos_func)
    assert value == approx(0.450183)

    res2 = binary_search(-1.0, 1.0, 1e-9, root=cos_func)
    assert res2 == approx(0.45018361)


def tan_func(x):
    return x - tan(x)


def test_tan_x():
    assert tan_func(0.5) == approx(-0.04630248)
    value = binary_search(2.0, 5.0, 1e-6, root=tan_func)
    assert value == approx(4.49341)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
