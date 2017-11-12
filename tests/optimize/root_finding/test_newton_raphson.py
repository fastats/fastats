from unittest import skip

from numpy import cos, tan, isnan
from pytest import approx, raises, warns

from fastats.optimise.root_finding import newton_raphson


def func(x):
    return x ** 3 - x - 1

nr_func = newton_raphson(1, 1e-6, root=func,
                         return_callable=True)


def test_basic_sanity():
    """
    Pedagogical example from S. Gourley's notes at
    University of Surrey.
    In this repo see `literature/NewtonRaphson.pdf`
    Online:
    http://personal.maths.surrey.ac.uk/st/S.Gourley/NewtonRaphson.pdf

    Solves x^3 - x - 1, starting at x0=1.0 to 5d.p.
    """
    no_kwargs = newton_raphson(1, 1e-6)
    assert no_kwargs == 0.0
    assert func(0.5) == approx(-1.375)

    value = nr_func(1, 1e-6)
    assert value == approx(1.324717)

    assert newton_raphson(1, 1e-6) == 0.0
    assert func(0.5) == approx(-1.375)


def test_delta_stops_early():
    value = nr_func(1, 1e-3)
    assert value == approx(1.324718)

    fixed = nr_func(1, 1e-6)
    assert fixed == approx(1.324717)


def test_multiple_x0():
    start_2 = nr_func(2.0, 1e-6)
    assert start_2 == approx(1.324717)

    start_1p5 = nr_func(1.5, 1e-6)
    assert start_1p5 == approx(1.324717)

    start_0p5 = nr_func(0.5, 1e-6)
    assert start_0p5 == approx(1.324717)


def test_with_local_function():
    """
    Example also taken from literature/NewtonRaphson.pdf
    """
    def cos_func(x):
        return cos(x) - 2 * x

    assert cos_func(0.5) == approx(-0.12241743)

    value = newton_raphson(0.5, 1e-6, root=cos_func)
    assert value == approx(0.4501836)

    assert cos_func(0.5) == approx(-0.12241743)


def tan_func(x):
    return x - tan(x)


def test_tan_x():
    """
    Example also taken from literature/NewtonRaphson.pdf
    """
    assert tan_func(0.5) == approx(-0.04630248)
    value = newton_raphson(4.6, 1e-6, root=tan_func)
    assert value == approx(4.49341)


def test_non_converging():
    """
    Example also taken from literature/NewtonRaphson.pdf
    """
    with raises(ZeroDivisionError): # Runtime Warning in python
        _ = newton_raphson(5.0, 1e-6, root=tan_func)

    # with warns(RuntimeWarning): # Runtime Warning in python
    #     _ = newton_raphson(5.0, 1e-6, root=tan_func)


if __name__ == '__main__':
    import pytest
    pytest.main()
