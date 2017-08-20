
import ast


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
    def __init__(self, change_params: dict, func_globals: dict, replaced: dict, new_funcs):
        self._params = change_params
        self._globals = func_globals
        self._replaced = replaced
        self._new_funcs = new_funcs

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if hasattr(node.func, 'attr'):
            # This will be hit where you have module
            # functions such as np.zeros_like
            return node

        name = node.func.id
        if name not in self._globals:
            # This will be hit for items not in the
            # function globals, such as `range`
            return node
        elif name in self._params:
            self._replaced[name] = self._globals[name]
            node = ast.copy_location(self.modify_function_name(node), node)
        else:
            # Lazy import because it's circular.
            from fastats.core.ast_transforms.processor import AstProcessor

            orig_inner_func = self._globals[node.func.id]
            self._replaced[node.func.id] = orig_inner_func
            proc = AstProcessor(orig_inner_func, self._params, self._replaced, self._new_funcs)
            new_inner_func = proc.process()
            self._globals[node.func.id] = new_inner_func
        ast.fix_missing_locations(node)
        return node

    def modify_function_name(self, call_node):
        new_name = self.new_name_from_call_name(call_node.func.id)
        call_node.func.id = new_name
        return call_node

    def new_name_from_call_name(self, call_name):
        func = self._params[call_name]
        return func.__name__
