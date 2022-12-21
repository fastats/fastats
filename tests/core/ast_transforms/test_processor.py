
import ast
from collections.abc import Iterable
from textwrap import dedent

from pytest import raises

from fastats.core.ast_transforms.processor import recompile, uncompile


def test_recompile_happy_path():
    func_as_string = dedent('''
    def f(n):
         # This is a comment
         return n ** 2
    ''')
    tree_module = ast.parse(func_as_string)
    recompile(tree_module, 'test_processor.py', 'exec')


def test_recompile_no_func():
    not_a_func = dedent('''
    12 + 12 * 3
    ''')
    tree_module = ast.parse(not_a_func)
    with raises(TypeError, match='Function body code not found'):
        recompile(tree_module, 'test_processor.py', 'exec')


def test_uncompile_happy_path():
    def sq(num):
        return num ** 2
    _ = sq(4)  # For test coverage metrics
    result = uncompile(sq.__code__)
    assert isinstance(result, Iterable)


def test_uncompile_lambdas():
    lam_square = lambda x: x ** 2
    _ = lam_square(5)  # For test coverage metrics
    with raises(TypeError, match='lambda functions not supported'):
        uncompile(lam_square.__code__)


def test_uncompile_string():
    compiled = compile('21 + 21', '<string>', 'exec')
    with raises(ValueError, match='code without source file not supported'):
        uncompile(compiled)


def test_uncompile_bad_file_path():
    compiled = compile('21 + 21', '/this/file/doesnt/exist.py', 'exec')
    with raises(Exception, match='source code not available'):
        uncompile(compiled)
