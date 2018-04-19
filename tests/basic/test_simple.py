
from pytest import approx

from fastats.basic import (
    zero, one, double, triple,
    square, cube,
    identity, flip, invert,
)

from fastats.testing.decorators import fast_no_fast


@fast_no_fast
def test_zero():
    assert zero() == 0
    assert zero() + 1.0 == 1.0


@fast_no_fast
def test_one():
    assert one() == 1
    assert one() + 2 == 3

    assert one() - one() == zero()


@fast_no_fast
def test_double():
    assert double(0) == 0
    assert double(2) == 4

    assert double(one()) == 2


@fast_no_fast
def test_triple():
    assert triple(0) == 0
    assert triple(-2) == -6

    assert triple(one()) == 3


@fast_no_fast
def test_square():
    assert square(0) == 0
    assert square(3) == 9

    assert square(double(one())) == 4


@fast_no_fast
def test_cube():
    assert cube(0) == 0
    assert cube(2) == 8

    assert cube(square(2)) == 64


@fast_no_fast
def test_identity():
    assert identity(0) == 0
    assert identity(1) == 1
    assert identity(-1) == -1
    assert identity(1.0) == 1.0


@fast_no_fast
def test_flip():
    assert flip(0) == 0
    assert flip(1) == -1

    assert flip(square(2)) == -4


@fast_no_fast
def test_invert():
    assert invert(1 / 2) == 2
    assert invert(2) == 1 / 2
    assert invert(-3) == approx(-1 / 3)

    assert flip(invert(-3)) == approx(1 / 3)
