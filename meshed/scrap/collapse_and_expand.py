"""Ideas on collapsing and expanding nodes
See "Collapse and expand nodes" discussion:
https://github.com/i2mint/meshed/discussions/54

"""

import re
from typing import Union, Iterable, Optional, Callable
from meshed.dag import DAG
from meshed.makers import code_to_dag


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


def get_src_string(src: Union[str, DAG]) -> str:
    if isinstance(src, str):
        return src
    elif hasattr(src, "_code_to_dag_src"):
        return get_src_string(src._code_to_dag_src)
    elif callable(src):
        import inspect

        return inspect.getsource(src)
    else:
        raise ValueError(
            f"src should be a string or have a _code_to_dag_src (meaning the src was "
            f"made with code_to_dag), not a {type(src)}"
        )


# TODO: Generalize to src that is any DAG
def collapse_function_calls(
    src: Union[str, DAG],
    call_func_name="call",
    *,
    rm_decorator="code_to_dag",
    include: Optional[Union[Iterable[str], Callable[[str], bool]]] = None,
):
    """
    Contract function calls in a source code string.

    That is, in source code, or a dag made from code_to_dag, replace calls of the form
    `call(func, arg)` with `func(arg)`.

    Note: Doesn't work with arbitrary DAG src, only those made from code_to_dag.
    """
    src_string = get_src_string(src)

    def should_include(func_name):
        if include is None:
            return True
        if isinstance(include, Iterable):
            return func_name in include
        if callable(include):
            return include(func_name)
        return False

    pattern = call_func_name + r"\(([^,]+),\s*([^)]+)\)"

    def replace(match):
        func_name, args = match.groups()
        if should_include(func_name):
            return f"{func_name}({args})"
        return match.group(0)

    new_src = re.sub(pattern, replace, src_string)

    if rm_decorator:
        new_src = remove_decorator_code(new_src, decorator_names=rm_decorator)

    return new_src if isinstance(src, str) else code_to_dag(new_src)


def expand_function_calls(
    src: Union[str, "DAG"],
    call_func_name="call",
    *,
    include: Optional[Union[Iterable[str], Callable[[str], bool]]] = None,
) -> str:
    """
    Inverse of collapse_function_calls.
    It replaces calls of the form `func(arg)` with `call(func, arg)`,
    except when the function call is part of a function definition header.
    If include is None, it expands all function calls.
    If include is a list of function names, only those functions are expanded.
    If include is a callable, it's used as a filter function.
    """
    src_string = get_src_string(src)

    def should_include(func_name):
        if include is None:
            return True
        if isinstance(include, Iterable):
            return func_name in include
        if callable(include):
            return include(func_name)
        return False

    pattern = r"(\b[a-zA-Z_]\w*)\(([^)]*)\)"

    def replace(match):
        # Get the start index of the match
        index = match.start()
        # Find the beginning of the current line
        line_start = src_string.rfind("\n", 0, index) + 1
        # Extract the text from the start of the line up to the match
        current_line = src_string[line_start:index]
        # If the current line starts with a function definition, skip expansion.
        if re.match(r"^\s*def\s", current_line):
            return match.group(0)
        func_name, args = match.groups()
        if should_include(func_name):
            return f"{call_func_name}({func_name}, {args})"
        return match.group(0)

    new_src = re.sub(pattern, replace, src_string)

    return new_src if isinstance(src, str) else code_to_dag(new_src)


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
