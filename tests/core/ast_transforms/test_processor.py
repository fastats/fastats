
import ast

import pytest

from fastats.core.ast_transforms.processor import recompile, uncompile


def test_recompile_happy_path():
    func_as_string = '''
def f(n):
     # This is a comment
     return n ** 2
'''
    tree_module = ast.parse(func_as_string)
    recompile(tree_module, 'test_processor.py', 'exec')


def test_recompile_no_func():
    not_a_func = '''
12 + 12 * 3
'''
    tree_module = ast.parse(not_a_func)
    with pytest.raises(TypeError, match='Function body code not found'):
        recompile(tree_module, 'test_processor.py', 'exec')


def test_uncompile_happy_path():
    def sq(num):
        return num ** 2
    _ = sq(4)  # For test coverage metrics
    result = uncompile(sq.__code__)
    try:
        iter(result)
    except Exception as err:
        raise Exception('''
        Expected the result of uncompile to be iterable, error below:
        {}'''.format(err))


def test_uncompile_lambdas():
    lam_square = lambda x: x ** 2
    _ = lam_square(5)  # For test coverage metrics
    with pytest.raises(TypeError, match='lambda functions not supported'):
        uncompile(lam_square.__code__)


def test_uncompile_string():
    compiled = compile('21 + 21', '<string>', 'exec')
    with pytest.raises(ValueError, match='code without source file not supported'):
        uncompile(compiled)


def test_uncompile_bad_file_path():
    compiled = compile('21 + 21', '/this/file/doesnt/exist.py', 'exec')
    with pytest.raises(Exception, match='source code not available'):
        uncompile(compiled)


if __name__ == 'main':
    pytest.main([__file__])
