"""Makers"""

from contextlib import suppress
from functools import partial
from typing import Mapping, Iterable, TypeVar, Callable
from itertools import product
from collections import defaultdict


T = TypeVar('T')

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


def edge_reversed_graph(
    g: Mapping[T, Iterable[T]],
    dst_nodes_factory: Callable[[], Iterable[T]] = list,
    dst_nodes_append: Callable[[Iterable[T], T], None] = list.append,
) -> Mapping[T, Iterable[T]]:
    """
    >>> g = dict(a='c', b='cd', c='abd', e='')
    >>> assert edge_reversed_graph(g) == {'c': ['a', 'b'], 'd': ['b', 'c'], 'a': ['c'], 'b': ['c'], 'e': []}
    >>> reverse_g_with_sets = edge_reversed_graph(g, set, set.add)
    >>> assert reverse_g_with_sets == {'c': {'a', 'b'}, 'd': {'b', 'c'}, 'a': {'c'}, 'b': {'c'}, 'e': set([])}

    Testing border cases
    >>> assert edge_reversed_graph(dict(e='', a='e')) == {'e': ['a'], 'a': []}
    >>> assert edge_reversed_graph(dict(a='e', e='')) == {'e': ['a'], 'a': []}
    """
    # Pattern: Groupby logic

    d = defaultdict(dst_nodes_factory)
    for src, dst_nodes in g.items():
        d.setdefault(src, dst_nodes_factory())  # add node if not present
        for dst in dst_nodes:  # empty iterable does nothing
            dst_nodes_append(d[dst], src)
    return d
