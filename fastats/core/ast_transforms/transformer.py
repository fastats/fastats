
import ast
from inspect import isbuiltin

import numpy as np

from fastats.core.ast_transforms.convert_to_jit import convert_to_jit


class CallTransform(ast.NodeTransformer):
    """
    The class which performs the AST mutations.

    This takes an ast tree and a dictionary of
    name -> function mappings, and recursively replaces
    any calls to `name` with the specified function.

    Warning
    -------
    `func_globals` and `replaced` are both **mutated**, not copied,
    by this class.

    `func_globals` is required as it's the `__globals__` dict on
    each function object. Unfortunately this dict can't be replaced
    (it is supposedly read-only), but given that it's a normal dict,
    it can be modified.

    Therefore during the `code_transform` context manager, we modify
    the globals dict with the ast-transformed function objects, then
    replace the original function objects in the context `exit` code.
    """
    def __init__(self, change_params: dict, func_globals: dict, replaced: dict, new_funcs: dict):
        self._params = change_params
        self._globals = func_globals
        self._replaced = replaced
        self._new_funcs = new_funcs

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if hasattr(node.func, 'attr'):
            # This will be hit where you have module
            # functions such as np.zeros_like.
            # We don't want fastats to modify these -
            # we see the module prefix as an indicator
            # that the author specifically wants to use
            # that function, so we just return early here.
            # To bypass this early return, import the
            # function and use without the module prefix, ie
            # from numpy import zeros_like
            # ...
            # a = zeros_like(x)
            return node

        name = node.func.id
        if name not in self._globals:
            # This will be hit for items not in the
            # function globals, such as `range`
            return node
        elif name in self._params:
            new_name = self.new_name_from_call_name(name)
            new_func = self._params[name]
            self._replaced[name] = self._globals[name]
            self._globals[name] = convert_to_jit(self._globals[name])
            self._globals[new_name] = convert_to_jit(new_func)

            new_node = ast.Call(
                func=ast.Name(id=new_name, ctx=ast.Load()),
                args=node.args, keywords=[]
            )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node
        else:
            # Lazy import because it's circular.
            from fastats.core.ast_transforms.processor import AstProcessor
            orig_inner_func = self._globals[node.func.id]

            not_ufunc = not isinstance(orig_inner_func, np.ufunc)
            not_builtin = not isbuiltin(orig_inner_func)
            if not_ufunc and not_builtin:
                self._replaced[node.func.id] = orig_inner_func
                proc = AstProcessor(
                    orig_inner_func, self._params,
                    self._replaced, self._new_funcs
                )
                new_inner_func = proc.process()
                self._globals[node.func.id] = convert_to_jit(new_inner_func)

        ast.fix_missing_locations(node)
        return node

    def new_name_from_call_name(self, call_name):
        func = self._params[call_name]
        return func.__name__
