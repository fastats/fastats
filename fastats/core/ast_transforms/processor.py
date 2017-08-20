"""
(1) Get kwargs from top-level func to override (now done in decorator)
(2) Work out what to do with each one (is it a func, a const etc)
(3) Replace each function call from leaf nodes upwards
"""

import ast
from copy import copy
import inspect
from inspect import signature, Parameter, isfunction, Signature
from pprint import pprint
import re
from types import CodeType

from fastats.core.ast_transforms.copy_func import copy_func
from fastats.core.ast_transforms.transformer import Transformer


class AstProcessor:
    def __init__(self, top_level_func, overrides, replaced):
        assert isfunction(top_level_func)

        self.top_level_func = copy_func(top_level_func)
        self._sig = signature(self.top_level_func)
        self._overrides = overrides
        self._replaced = replaced
        self._debug = self._overrides.get('debug')

    def process(self):
        print('Globs: ', self.top_level_func, id(self.top_level_func.__globals__))

        source = inspect.getsource(self.top_level_func)
        tree = ast.parse(source)
        globs = self.top_level_func.__globals__
        t = Transformer(self._overrides, globs, self._replaced)
        new_tree = t.visit(tree)

        # TODO remove the fs decorator from within the ast code
        new_tree.body[0].decorator_list = []
        ast.fix_missing_locations(new_tree)
        if self._debug:
            pprint(ast.dump(new_tree))

        code_obj = self.recompile(new_tree, '<fastats>', 'exec')

        func_copy = copy(self.top_level_func)
        func_copy.__code__ = code_obj
        print('Copy: ', self.top_level_func, id(func_copy.__globals__))
        return func_copy

    def recompile(self, source, filename, mode, flags=0, firstlineno=1, privateprefix=None):
        """ recompile output of uncompile back to a code object. source may also be preparsed AST """
        a = source
        node = a.body[0]
        if not isinstance(node, ast.FunctionDef):
            raise TypeError('Expecting function AST node')

        c0 = compile(a, filename, mode, flags, True)

        # This code object defines the function. Find the function's actual body code:
        for c in c0.co_consts:
            if not isinstance(c, CodeType):
                continue
            if c.co_name == node.name and c.co_firstlineno == node.lineno:
                break
        else:
            raise TypeError('Function body code not found')

        # Re-mangle private names:
        if privateprefix is not None:
            def fixnames(names):
                isprivate = re.compile('^__.*(?<!__)$').match
                return tuple(privateprefix + name if isprivate(name) else name for name in names)

            c = CodeType(c.co_argcount, c.co_nlocals, c.co_stacksize, c.co_flags, c.co_code, c.co_consts,
                     fixnames(c.co_names), fixnames(c.co_varnames), c.co_filename, c.co_name,
                     c.co_firstlineno, c.co_lnotab, c.co_freevars, c.co_cellvars)
        return c