
import ast
import inspect
import re
from inspect import signature
from pprint import pprint
from types import CodeType

from numba import jit

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit
from fastats.core.ast_transforms.copy_func import copy_func
from fastats.core.ast_transforms.transformer import CallTransform


class AstProcessor:
    def __init__(self, top_level_func, overrides, replaced, new_funcs=None):
        self.top_level_func = copy_func(top_level_func, new_funcs or {})
        self._new_funcs = new_funcs or {}
        self._sig = signature(self.top_level_func)
        self._overrides = overrides
        self._replaced = replaced
        self._debug = self._overrides.get('debug')

    def process(self):
        try:
            source = inspect.getsource(self.top_level_func)
            tree = ast.parse(source)
        except IndentationError:
            data = uncompile(self.top_level_func.__code__)
            tree = parse_snippet(*data)

        globs = self.top_level_func.__globals__
        globs['jit'] = jit
        t = CallTransform(self._overrides, globs, self._replaced, self._new_funcs)
        new_tree = t.visit(tree)

        # TODO remove the fs decorator from within the ast code
        new_tree.body[0].decorator_list = [ast.Name(id='jit', ctx=ast.Load())]
        ast.fix_missing_locations(new_tree)
        if self._debug:
            pprint(ast.dump(new_tree))

        code_obj = self.recompile(new_tree, '<fastats>', 'exec', globs=globs)

        self.top_level_func.__code__ = code_obj
        return convert_to_jit(self.top_level_func)

    def recompile(self, source, filename, mode, flags=0, privateprefix=None, globs=None):
        """
        This is based on an ActiveState recipe by Oren Tirosh:
        http://code.activestate.com/recipes/578353-code-to-source-and-back/

        Recompiles output back to a code object.
        Source may also be preparsed AST.
        """
        node = source.body[0]

        c0 = compile(source, filename, mode, flags, True)

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
                return tuple(privateprefix + name if isprivate(name)
                             else name for name in names)

            c = CodeType(
                c.co_argcount, c.co_nlocals, c.co_stacksize, c.co_flags,
                c.co_code, c.co_consts, fixnames(c.co_names),
                fixnames(c.co_varnames), c.co_filename, c.co_name,
                c.co_firstlineno, c.co_lnotab, c.co_freevars, c.co_cellvars)
        return c


def uncompile(c):
    """ uncompile(codeobj) -> [source, filename, mode, flags, firstlineno, privateprefix] """
    # if c.co_flags & inspect.CO_NESTED or c.co_freevars:
    #     raise Exception('nested functions not supported')
    if c.co_name == '<lambda>':
        raise Exception('lambda functions not supported')
    if c.co_filename == '<string>':
        raise Exception('code without source file not supported')

    filename = inspect.getfile(c)
    try:
        lines, firstlineno = inspect.getsourcelines(c)
    except IOError:
        raise Exception('source code not available')
    source = ''.join(lines)

    # __X is mangled to _ClassName__X in methods. Find this prefix:
    privateprefix = None
    for name in c.co_names:
        m = re.match('^(_[A-Za-z][A-Za-z0-9_]*)__.*$', name)
        if m:
            privateprefix = m.group(1)
            break

    return [source, filename, 'exec', c.co_flags, firstlineno, privateprefix]


def parse_snippet(source, filename, mode, flags, firstlineno, privateprefix_ignored=None):
    """ Like ast.parse, but accepts indented code snippet with a line number offset. """
    args = filename, mode, ast.PyCF_ONLY_AST, True
    prefix = '\n'
    try:
        a = compile(prefix + source, *args)
    except IndentationError:
        # Already indented? Wrap with dummy compound statement
        prefix = 'with 0:\n'
        a = compile(prefix + source, *args)
        # peel wrapper
        a.body = a.body[0].body
    ast.increment_lineno(a, firstlineno - 2)
    return a
