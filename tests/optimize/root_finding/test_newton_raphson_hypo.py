
from hypothesis import given, assume
from hypothesis.strategies import (
    tuples, integers, floats
)
from numpy import cos
from pytest import approx

from fastats.optimise.root_finding import newton_raphson


def func(x):
    return x**3 - x - 1


def less_or_equal(x, compared_to, rel=1e-6):
    return ((x < compared_to)
            or ((x - compared_to) == approx(0.0, rel=rel))
            or (x == approx(x, rel=rel)))

nr_func = newton_raphson(1, 1e-6, root=func, return_callable=True)


@given(floats(min_value=0.01, max_value=3.5))
def test_minimal(x):
    """
    Tests that the value output from the solver
    is less than or equal to the value of the
    objective.
    """
    eps = 1e-12
    value = nr_func(x, eps)

    assume(func(x) > 0.0)

    assert less_or_equal(value, compared_to=func(x))


def cos_func(x):
    return cos(x) - 2 * x

nr_cos = newton_raphson(0.5, 1e-6, root=cos_func, return_callable=True)


@given(floats(min_value=0.3, max_value=0.8))
def test_cos_minus_2x(x):
    value = nr_cos(x, 1e-6)
    assert less_or_equal(value, compared_to=cos_func(x))


if __name__ == '__main__':
    import pytest
    pytest.main()
