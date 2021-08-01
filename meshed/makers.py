"""Makers"""

from contextlib import suppress
from typing import Mapping, Iterable, TypeVar, Callable
from itertools import product
from collections import defaultdict


T = TypeVar("T")

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
