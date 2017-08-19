"""
Update any functions in the tree with the args
specified.
"""

from logging import info
import ast


class Transformer(ast.NodeTransformer):
    def __init__(self, change_params: dict, func_globals: dict, replaced: dict):
        self._params = change_params
        self._globals = func_globals
        self._replaced = replaced
        # super().__init__()

    def visit_Call(self, node):
        print('Call: ', node.func.id)
        node = self.generic_visit(node)
        name = node.func.id

        if name in self._params:
            self._replaced[name] = self._globals[name]
            node = ast.copy_location(self.modify_function_name(node), node)
        else:
            # Circular import.
            from fastats.core.ast_transforms.processor import AstProcessor

            orig_inner_func = self._globals[node.func.id]
            self._replaced[node.func.id] = orig_inner_func
            proc = AstProcessor(orig_inner_func, self._params, self._replaced)
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
        return func.__name__  # TODO : do this properly.
