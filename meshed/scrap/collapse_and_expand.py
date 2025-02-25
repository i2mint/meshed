"""Ideas on collapsing and expanding nodes
See "Collapse and expand nodes" discussion: 
https://github.com/i2mint/meshed/discussions/54

"""

import re
from typing import Union, Iterable, Optional


def remove_decorator_code(
    src: str, decorator_names: Optional[Union[str, Iterable[str]]] = None
) -> str:
    """
    Remove the code corresponding to decorators from a source code string.
    If decorator_names is None, will remove all decorators.
    If decorator_names is an iterable of strings, will remove the decorators with those names.

    Examples:
    >>> src = '''
    ... @decorator
    ... def func():
    ...     pass
    ... '''
    >>> print(remove_decorator_code(src))
    def func():
        pass

    >>> src = '''
    ... @decorator1
    ... @decorator2
    ... def func():
    ...     pass
    ... '''
    >>> print(remove_decorator_code(src, "decorator1"))
    @decorator2
    def func():
        pass
    """
    import ast

    if isinstance(decorator_names, str):
        decorator_names = [decorator_names]

    class DecoratorRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.decorator_list:
                if decorator_names is None:
                    node.decorator_list = []  # Remove all decorators
                else:
                    node.decorator_list = [
                        d
                        for d in node.decorator_list
                        if not (isinstance(d, ast.Name) and d.id in decorator_names)
                    ]
            return node

        def visit_ClassDef(self, node):
            if node.decorator_list:
                if decorator_names is None:
                    node.decorator_list = []  # Remove all decorators
                else:
                    node.decorator_list = [
                        d
                        for d in node.decorator_list
                        if not (isinstance(d, ast.Name) and d.id in decorator_names)
                    ]
            return node

    tree = ast.parse(src)
    new_tree = DecoratorRemover().visit(tree)
    ast.fix_missing_locations(new_tree)

    return ast.unparse(new_tree)


# TODO: Generalize to src that is any DAG
def contract_function_calls(
    src: str, call_func_name='call', *, rm_decorator='code_to_dag'
):
    """
    Contract function calls in a source code string.

    That is, in source code, or a dag made from code_to_dag, replace calls of the form
    `call(func, arg)` with `func(arg)`.

    Note: Doesn't work with arbitrary DAG src, only those made from code_to_dag.
    """
    if hasattr(src, '_code_to_dag_src'):
        import inspect

        dag_of_code_to_dag = src
        src_string = inspect.getsource(dag_of_code_to_dag._code_to_dag_src)
    elif isinstance(src, str):
        src_string = src
    else:
        raise ValueError(
            f"src should be a string or have a _code_to_dag_src (meaning the src was "
            f"made with code_to_dag), not a {type(src)}"
        )

    new_src = re.sub(call_func_name + r'\(([^,]+),\s*([^)]+)\)', r'\1(\2)', src_string)
    if rm_decorator:
        # Remove the docorator code
        # TODO: If the decorator has arguments, we'll lose those here
        new_src = remove_decorator_code(new_src, decorator_names=rm_decorator)

    if isinstance(src, str):
        return new_src
    else:
        # TODO: If the decorator has arguments, this will not restore the original code
        return code_to_dag(new_src)


# ------------------------------------------------------------------------------
# Older code

from dataclasses import dataclass
from i2 import Sig
from meshed.dag import DAG


@dataclass
class CollapsedDAG:
    """To collapse a DAG into a single function

    This is useful for when you want to use a DAG as a function,
    but you don't want to see all the arguments.

    """

    dag: DAG

    def __post_init__(self):
        Sig(self.dag)(self)  # so that __call__ gets dag's signature
        self.__name__ = self.dag.name

    def __call__(self, *args, **kwargs):
        return self.dag(*args, **kwargs)

    def expand(self):
        return self.dag


# TODO: Finish this
def expand_nodes(
    dag,
    nodes=None,
    *,
    is_node=lambda fnode, node: fnode.name == node or fnode.out == node,
    expansion_record_store=None,  # TODO: Implement this to keep track of what was expanded
):
    if nodes is None:
        nodes = ...  # find all func_nodes that have isinstance(fn.func, CollapsedDAG)

    def change_node_or_not(node):
        if is_node(node):
            return CollapsedDAG(node.func)
        else:
            return node

    return DAG(list(map(change_node_or_not, dag.func_nodes)))
