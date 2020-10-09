from typing import Mapping, Any, Sized, Callable
from itertools import product
from collections import defaultdict
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


def edge_reversed_graph(g: Mapping[Any, Sized]):
    """
    >>> g = dict(a='c', b='cd', c='abd', e='')
    >>> assert edge_reversed_graph(g) == {'c': {'a', 'b'}, 'd': {'c', 'b'}, 'a': {'c'}, 'b': {'c'}, 'e': set()}
    """
    # Pattern: Groupby logic
    d = defaultdict(set)
    for src, dst_nodes in g.items():
        if len(dst_nodes):
            for dst in dst_nodes:
                d[dst].add(src)
        else:
            d[src] = set()
    return d
