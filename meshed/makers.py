"""Makers"""

from contextlib import suppress
from functools import partial
from typing import Mapping, Iterable, TypeVar, Callable
from meshed import FuncNode
from i2 import Pipe
import ast
import inspect

T = TypeVar('T')

# Some restrictions exist and need to be clarified or removed (i.e. more cases handled)
# For example,
# * tuple assignment not handled (x, y = func(...))
# * can't reuse a variable (would lead to same node)
# * x = y (or x, y = tup) not handled (but could easily by binding)
# We don't need these cases to be handled, only x = func(...) forms lead to Turing (I
# think...):
# Further other cases are not handled, but we don't want to handle ALL of python
# -- just a sufficiently expressive subset.

import ast
import inspect
from typing import Tuple, Callable


def attr_dict(obj):
    return {a: getattr(obj, a) for a in dir(obj) if not a.startswith('_')}


def is_from_ast_module(o):
    return getattr(type(o), '__module__', '').startswith('_ast')


def _ast_info_str(x):
    return f'lineno={x.lineno}'


# Note: generalize? glom?
def parse_assignment(body: ast.Assign) -> Tuple:
    # TODO: Make this validation better (at least more help in raised error)
    # TODO: extract validating code out as validation functions?
    info = _ast_info_str(body)
    if not isinstance(body, ast.Assign):
        raise ValueError(f"All commands should be assignments, this one wasn't: {info}")

    target = body.targets
    assert len(target) == 1, f'Only one target allowed: {info}'
    target = target[0]
    assert isinstance(
        target, (ast.Name, ast.Tuple)
    ), f'Should be a ast.Name or ast.Tuple: {info}'

    value = body.value
    assert isinstance(value, ast.Call), f'Only one target allowed: {info}'

    return target, value


def parsed_to_node_kwargs(target_value) -> dict:
    """Extract FuncNode kwargs (name, out, and bind) from ast (target,value) pairs

    :param target_value: A (target, value) pair
    :return: A ``{name:..., out:..., bind:...}`` dict (meant to be used to curry FuncNode

    """
    # Note: ast.Tuple has names in 'elts' attribute,
    # and could be handled, but would need to lead to multiple nodes
    target, value = target_value
    assert isinstance(target, ast.Name), f'Should be a ast.Name: {target}'
    args = value.args
    bind_from_args = {i: k.id for i, k in enumerate(args)}
    kwargs = {x.arg: x.value.id for x in value.keywords}
    return dict(name=value.func.id, out=target.id, bind=dict(bind_from_args, **kwargs))


FuncNodeFactory = Callable[[Callable], FuncNode]


def node_kwargs_to_func_node_factory(node_kwargs) -> FuncNodeFactory:
    return partial(FuncNode, **node_kwargs)


class AssignNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.store = []

    def visit_Assign(self, node):
        self.store.append(parse_assignment(node))
        return node


def retrieve_assignments(src):
    if callable(src):
        src = inspect.getsource(src)
    nodes = ast.parse(src)
    visitor = AssignNodeVisitor()
    visitor.visit(nodes)

    return visitor.store


def parse_assignment_steps(src):
    if callable(src):
        src = inspect.getsource(src)
    root = ast.parse(src)
    assert len(root.body) == 1
    func_body = root.body[0]
    # TODO, work with func_body.args to get info on interface (name, args, kwargs,
    #  return etc.)
    #     return func_body
    for body in func_body.body:
        yield parse_assignment(body)


iterize = lambda func: partial(map, func)

targval_to_func_node_factory = Pipe(
    parsed_to_node_kwargs, node_kwargs_to_func_node_factory
)
src_to_func_node_factory = Pipe(
    parse_assignment_steps, iterize(targval_to_func_node_factory)
)

# TODO: Rewrite body to use ast tools above!
# TODO: This function should really be a DAG where we can choose if we want parsed lines,
#   digraph dot commands, the graphviz.Digraph object itself etc.
# TODO: The function below is meant to evolve into a tool that can take python code,
#  (possibly written with some style constraints) and produce an equivalent DAG.
#   Here, it just produces a graphviz.Digraph of the code.
def simple_code_to_digraph(code):
    """Make a graphviz.Digraph object based on code (string or function's body)"""
    import re
    from inspect import getsource
    from graphviz import Digraph

    def get_code_str(code) -> str:
        if not isinstance(code, str):
            return getsource(code)
        return code

    empty_spaces = re.compile('^\s*$')
    simple_assignment_p = re.compile(
        '(?P<output_vars>[^=]+)' '\s*=\s*' '(?P<func>\w+)' '\((?P<input_vars>.*)\)'
    )

    def get_lines(code_str):
        for line in code_str.split('\n'):
            if not empty_spaces.match(line):
                yield line.strip()

    def groupdict_parser(s, pattern):
        pattern = re.compile(pattern)
        m = pattern.search(s)
        if m:
            return m.groupdict()
        return None

    def parsed_lines(
        code_str, line_parser=partial(groupdict_parser, pattern=simple_assignment_p)
    ):
        yield from filter(None, map(line_parser, get_lines(code_str)))

    def parsed_lines_to_dot(parsed_lines):
        for d in parsed_lines:
            yield f"{d['func']} [shape=box]"
            yield f"{d['input_vars']} -> {d['func']} -> {d['output_vars']}"

    code_str = get_code_str(code)
    dot_lines = parsed_lines_to_dot(parsed_lines(code_str))
    return Digraph(body=dot_lines)


# TODO: Replace using builtin random
with suppress(ModuleNotFoundError, ImportError):
    from numpy.random import randint, choice

    def random_graph(n_nodes=7):
        """Get a random graph"""
        nodes = range(n_nodes)

        def gen():
            for src in nodes:
                n_dst = randint(0, n_nodes - 1)
                dst = choice(n_nodes, n_dst, replace=False)
                yield src, list(dst)

        return dict(gen())
