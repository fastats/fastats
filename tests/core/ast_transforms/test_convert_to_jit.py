
import math
from functools import partial

from numba import jit
from pytest import raises

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit


def test_doesnt_convert_math_builtins():
    for func in (math.atan2, math.atanh, math.degrees, math.exp, math.floor, math.log,
                 math.sin, math.sinh, math.tan, math.tanh):
        assert convert_to_jit(func) is func


def test_doesnt_convert_jitted_functions():
    @jit
    def jit_func():
        return 5

    assert convert_to_jit(jit_func) is jit_func


def test_doesnt_convert_converted_functions():
    def jit_func():
        return 5

    converted = convert_to_jit(jit_func)

    assert convert_to_jit(converted) is converted


def test_converts_simple_function():
    def add(a, b):
        return a + b

    jitted = convert_to_jit(add)

    assert jitted(1, 2) == 3


def test_raises_for_non_func():
    with raises(TypeError):
        convert_to_jit('And Now for Something Completely Different')

    with raises(TypeError):
        convert_to_jit({'answer': 42})

    callable_but_not_function = partial(sum)

    with raises(TypeError):
        convert_to_jit(callable_but_not_function)
