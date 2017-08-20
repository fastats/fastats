
from numpy import cos, tan, isnan
from pytest import approx, raises

from fastats.optimise.root_finding import newton_raphson


def func(x):
    return x ** 3 - x - 1


def test_basic_sanity():
    """
    Pedagogical example from S. Gourley's notes at
    University of Surrey.
    In this repo see `literature/NewtonRaphson.pdf`
    Online:
    http://personal.maths.surrey.ac.uk/st/S.Gourley/NewtonRaphson.pdf

    Solves x^3 - x - 1, starting at x0=1.0 to 5d.p.
    """
    value = newton_raphson(1, 1e-6, root=func)
    assert value == approx(1.324717)


def test_delta_stops_early():
    value = newton_raphson(1, 1e-3, root=func)
    assert value == approx(1.324718)

    fixed = newton_raphson(1, 1e-6, root=func)
    assert fixed == approx(1.324717)


def test_multiple_x0():
    start_2 = newton_raphson(2.0, 1e-6, root=func)
    assert start_2 == approx(1.324717)

    start_1p5 = newton_raphson(1.5, 1e-6, root=func)
    assert start_1p5 == approx(1.324717)

    start_0p5 = newton_raphson(0.5, 1e-6, root=func)
    assert start_0p5 == approx(1.324717)


def test_with_local_function():
    """
    Example also taken from literature/NewtonRaphson.pdf
    """
    def cos_func(x):
        return cos(x) - 2 * x

    value = newton_raphson(0.5, 1e-6, root=cos_func)
    assert value == approx(0.4501836)


def test_tan_x():
    """
    Example also taken from literature/NewtonRaphson.pdf
    """
    def tan_func(x):
        return x - tan(x)

    value = newton_raphson(4.6, 1e-6, root=tan_func)
    assert value == approx(4.49341)


def test_non_converging():
    """
    Example also taken from literature/NewtonRaphson.pdf
    """
    def tan_func(x):
        return x - tan(x)

    value = newton_raphson(5.0, 1e-6, root=tan_func)
    assert isnan(value)


if __name__ == '__main__':
    import pytest
    pytest.main()
