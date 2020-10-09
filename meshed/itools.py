"""Functions that provide iterators of g elements where g is any adjacency Mapping representation.

"""
from typing import Any, Mapping, Sized
from itertools import product
from meshed.makers import edge_reversed_graph


def edges(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(edges(g))
    [('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'), ('d', 'c'), ('e', 'c'), ('e', 'z')]
    """
    for src in g:
        for dst in g[src]:
            yield src, dst


def nodes(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(nodes(g))
    ['a', 'b', 'c', 'd', 'e', 'f', 'z']
    """
    seen = set()
    for src in g:
        if src not in seen:
            yield src
            seen.add(src)
        for dst in g[src]:
            if dst not in seen:
                yield dst
                seen.add(dst)


def source_nodes(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(source_nodes(g))
    ['a', 'b', 'c', 'd', 'e', 'f']

    Note `f` is there. It's a strange one because it is neither the source of any destination, nor the destination
    of any source. But since it's listed in the dict, we'll consider it as a source.
    """
    yield from g


# Note: `yield from set(dst for src in g for dst in g[src])` might be simpler, but 25% slower.
#   but just taking `set(dst for src in g for dst in g[src])` (not in a function that yields from...)
#   is actually 25% FASTER.
def dest_nodes(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(dest_nodes(g))
    ['a', 'b', 'c', 'd', 'e', 'z']
    """
    seen = set()
    for src in g:
        for dst in g[src]:
            if dst not in seen:
                yield dst
                seen.add(dst)


def isolated_nodes(g: Mapping):
    """Nodes that
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> set(isolated_nodes(g))
    {'f'}
    """
    for src in g:
        if not next(iter(g[src]), False):  # Note: Slower than just `not g[src]`, but safer
            yield src


def find_path(g: Mapping, src, dst, path=None):
    """ find a path from src to dst nodes in graph

    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> find_path(g, 'a', 'c')
    ['a', 'c']
    >>> find_path(g, 'a', 'b')
    ['a', 'c', 'b']
    >>> find_path(g, 'a', 'z')
    ['a', 'c', 'b', 'e', 'z']
    >>> assert find_path(g, 'a', 'f') == None

    """
    if path == None:
        path = []
    path = path + [src]
    if src == dst:
        return path
    if src not in g:
        return None
    for node in g[src]:
        if node not in path:
            extended_path = find_path(g, node, dst, path)
            if extended_path:
                return extended_path
    return None


def reverse_edges(g: Mapping):
    for src, dst_nodes in g.items():
        yield from product(dst_nodes, src)


def out_degrees(g: Mapping[Any, Sized]):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> dict(out_degrees(g))
    {'a': 1, 'b': 2, 'c': 4, 'd': 1, 'e': 2, 'f': 0}
    """
    for src, dst_nodes in g.items():
        yield src, len(dst_nodes)


def in_degrees(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> dict(in_degrees(g))
    {'a': 1, 'b': 2, 'c': 4, 'd': 1, 'e': 2, 'f': 0}
    """
    return out_degrees(edge_reversed_graph(g))
