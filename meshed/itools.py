"""Functions that provide iterators of graph elements where graph is any adjacency Mapping representation.

"""
from collections.abc import Mapping


def edges(graph: Mapping):
    """
    >>> graph = dict(a='c', b='ce', c='abde', d='c', e=['c', 'b'], f={})
    >>> sorted(edges(graph))
    [('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'), ('d', 'c'), ('e', 'b'), ('e', 'c')]
    """
    for from_node in graph:
        for to_node in graph[from_node]:
            yield from_node, to_node


def nodes(graph: Mapping):
    """
    >>> graph = dict(a='c', b='ce', c='abde', d='c', e=['c', 'b'], f={})
    >>> sorted(nodes(graph))
    ['a', 'b', 'c', 'd', 'e', 'f']
    """
    nodes_already_seen = set()
    for from_node in graph:
        if from_node not in nodes_already_seen:
            yield from_node
            nodes_already_seen.add(from_node)
        for to_node in graph[from_node]:
            if to_node not in nodes_already_seen:
                yield to_node
                nodes_already_seen.add(to_node)


def isolated_nodes(graph: Mapping):
    """
    >>> graph = dict(a='c', b='ce', c='abde', d='c', e=['c', 'b'], f={})
    >>> set(isolated_nodes(graph))
    {'f'}
    """
    for from_node in graph:
        if not graph[from_node]:
            yield from_node
