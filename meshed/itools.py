"""Functions that provide iterators of g elements where g is any
adjacency Mapping representation.

"""
from typing import Any, Mapping, Sized, MutableMapping, Iterable, Callable, TypeVar
from itertools import product, chain
from functools import wraps
from collections import defaultdict

from i2.signatures import Sig


def _handle_exclude_nodes(func):
    sig = Sig(func)

    @wraps(func)
    def _func(*args, **kwargs):
        kwargs = sig.kwargs_from_args_and_kwargs(args, kwargs, apply_defaults=True)
        try:
            _exclude_nodes = kwargs['_exclude_nodes']
        except KeyError:
            raise RuntimeError(f"{func} doesn't have a _exclude_nodes argument")

        if _exclude_nodes is None:
            _exclude_nodes = set()
        elif not isinstance(_exclude_nodes, set):
            _exclude_nodes = set(_exclude_nodes)

        kwargs['_exclude_nodes'] = _exclude_nodes
        args, kwargs = sig.args_and_kwargs_from_kwargs(kwargs)
        return func(*args, **kwargs)

    return _func


def add_edge(g: MutableMapping, node1, node2):
    """Add an edge FROM node1 TO node2"""
    if node1 in g:
        g[node1].append(node2)
    else:
        g[node1] = [node2]


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


def has_node(g: Mapping, node, check_adjacencies=True):
    """Returns True if the graph has given node

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2]
    ... }
    >>> has_node(g, 0)
    True
    >>> has_node(g, 2)
    True

    Note that 2 was found, though it's not a key of ``g``.
    This shows that we don't have to have an explicit ``{2: []}`` in ``g``
    to be able to see that it's a node of ``g``.
    The function will go through the values of the mapping to try to find it
    if it hasn't been found before in the keys.

    This can be inefficient, so if that matters, you can express your
    graph ``g`` so that all nodes are explicitly declared as keys, and
    use ``check_adjacencies=False`` to tell the function not to look into
    the values of the ``g`` mapping.

    >>> has_node(g, 2, check_adjacencies=False)
    False
    >>> g = {
    ...     0: [1, 2],
    ...     1: [2],
    ...     2: []
    ... }
    >>> has_node(g, 2, check_adjacencies=False)
    True

    """
    if node in g:
        return True

    if check_adjacencies:
        # look in the adjacencies (because a leaf might not show up as a
        # {leaf: []} item in g!
        for adjacencies in g.values():
            if node in adjacencies:
                return True

    return False  # if not found before


@_handle_exclude_nodes
def successors(g: Mapping, node, _exclude_nodes=None):
    """Iterator of nodes that have directed paths FROM node

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]}
    >>> assert set(successors(g, 1)) == {1, 2, 3, 4}
    >>> assert set(successors(g, 3)) == {4}
    >>> assert set(successors(g, 4)) == set()

    Notice that 1 is a successor of 1 here because there's a 1-2-1 directed path
    """
    direct_successors = set(g.get(node, [])) - _exclude_nodes
    yield from direct_successors
    _exclude_nodes.update(direct_successors)
    for successor_node in direct_successors:
        yield from successors(g, successor_node, _exclude_nodes)


def predecessors(g: Mapping, node):
    """Iterator of nodes that have directed paths TO node

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]}
    >>> set(predecessors(g, 4))
    {0, 1, 2, 3}
    >>> set(predecessors(g, 2))
    {0, 1, 2}
    >>> set(predecessors(g, 0))
    set()

    Notice that 2 is a predecessor of 2 here because of the presence
    of a 2-1-2 directed path.
    """
    yield from successors(edge_reversed_graph(g), node)


def _split_if_str(x):
    """
    If source is a string, the `str.split()` of it will be returned.

    This is to be used in situations where we deal with lists of strings and
    want to avoid mistaking a single string input with an iterable of characters.

    For example, if a user specifies ``'abc'`` as an argument, this could have the
    same effect as specifying  ``['a', 'b', 'c']``,
    which often not what's intended, but rather ``['abc']`` is intended).

    If the user actually wants ``['a', 'b', 'c']``, they can specify it by doing
    ``list('abc')`` explicitly.
    """
    if isinstance(x, str):
        return x.split()
    else:
        return x


def children(g: Mapping, source: Iterable):
    """Set of all nodes (not in source) adjacent FROM 'source' in 'g'

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]
    ... }
    >>> children(g, [2, 3])
    {1, 4}
    >>> children(g, [4])
    set()
    """
    source = _split_if_str(source)
    source = set(source)
    _children = set()
    for node in source:
        _children.update(g.get(node, set()))
    return _children - source


def parents(g: Mapping, source: Iterable):
    """Set of all nodes (not in source) adjacent TO 'source' in 'g'

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]
    ... }
    >>> parents(g, [2, 3])
    {0, 1}
    >>> parents(g, [0])
    set()
    """
    return children(edge_reversed_graph(g), source)


@_handle_exclude_nodes
def ancestors(g: Mapping, source: Iterable, _exclude_nodes=None):
    """Set of all nodes (not in source) reachable TO `source` in `g`.

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [4],
    ...     3: [4]
    ... }
    >>> ancestors(g, [2, 3])
    {0, 1}
    >>> ancestors(g, [0])
    set()
    """
    source = _split_if_str(source)
    assert isinstance(source, Iterable)
    source = set(source) - _exclude_nodes
    _parents = (set(parents(g, source)) - source) - _exclude_nodes
    if not _parents:
        return set()
    else:
        _ancestors_of_parent = ancestors(g, _parents, _exclude_nodes)

        return _parents | _ancestors_of_parent


def descendants(g: Mapping, source: Iterable, _exclude_nodes=None):
    """Returns the set of all nodes reachable FROM `source` in `g`.

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [4],
    ...     3: [4]
    ... }
    >>> descendants(g, [2, 3])
    {4}
    >>> descendants(g, [4])
    set()
    """
    return ancestors(edge_reversed_graph(g), source, _exclude_nodes)


# TODO: Can serious be optimized, and hasn't been tested much: Revise
def root_nodes(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(root_nodes(g))
    ['f']

    Note that `f` is present: Isolated nodes are considered both as
    root and leaf nodes both.
    """
    nodes_having_parents = set(chain.from_iterable(g.values()))
    return set(g) - set(nodes_having_parents)


# TODO: Can serious be optimized, and hasn't been tested much: Revise
def leaf_nodes(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(leaf_nodes(g))
    ['f', 'z']

    Note that `f` is present: Isolated nodes are considered both as
    root and leaf nodes both.
    """
    return root_nodes(edge_reversed_graph(g))


def isolated_nodes(g: Mapping):
    """Nodes that
    >>> g = dict(a='c', b='ce', c=list('abde'), d='c', e=['c', 'z'], f={})
    >>> set(isolated_nodes(g))
    {'f'}
    """
    for src in g:
        if not next(
            iter(g[src]), False
        ):  # Note: Slower than just `not g[src]`, but safer
            yield src


def find_path(g: Mapping, src, dst, path=None):
    """find a path from src to dst nodes in graph

    >>> g = dict(a='c', b='ce', c=list('abde'), d='c', e=['c', 'z'], f={})
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
    >>> assert dict(out_degrees(g)) == (
    ...     {'a': 1, 'b': 2, 'c': 4, 'd': 1, 'e': 2, 'f': 0}
    ... )
    """
    for src, dst_nodes in g.items():
        yield src, len(dst_nodes)


def in_degrees(g: Mapping):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert dict(in_degrees(g)) == (
    ... {'a': 1, 'b': 1, 'c': 4,  'd': 1, 'e': 2, 'f': 0, 'z': 1}
    ... )
    """
    return out_degrees(edge_reversed_graph(g))


def copy_of_g_with_some_keys_removed(g: Mapping, keys: Iterable):
    keys = _split_if_str(keys)
    return {k: v for k, v in g.items() if k not in keys}


def _topological_sort_helper(g, v, visited, stack):
    """A recursive function to service topological_sort"""

    visited.add(v)  # Mark the current node as visited.

    # Recurse for all the vertices adjacent to this node
    for i in g.get(v, []):
        if i not in visited:
            _topological_sort_helper(g, i, visited, stack)

    # Push current node to stack which stores result
    stack.insert(0, v)


def topological_sort(g: Mapping):
    """Return the list of nodes in topological sort order.

    This order is such that a node parents will all occur before;
        If order[i] is parent of order[j] then i < j

    This is often used to compute the order of computation in a DAG.

    >>> g = {
    ...     0: [4, 2],
    ...     4: [3, 1],
    ...     2: [3],
    ...     3: [1]
    ... }
    >>>
    >>> list(topological_sort(g))
    [0, 2, 4, 3, 1]

    Here's an ascii art of the graph, to verify that the topological sort is
    indeed as expected.

    .. code-block::
    ┌───┐     ┌───┐     ┌───┐     ┌───┐
    │ 0 │ ──▶ │ 2 │ ──▶ │ 3 │ ──▶ │ 1 │
    └───┘     └───┘     └───┘     └───┘
      │                   ▲         ▲
      │                   │         │
      ▼                   │         │
    ┌───┐                 │         │
    │ 4 │ ────────────────┼─────────┘
    └───┘                 │
      │                   │
      └───────────────────┘
    """
    visited = set()
    stack = []

    # Call the recursive helper function to accumulate topological sorts
    # starting from all vertices one by one
    for i in g:
        if i not in visited:
            _topological_sort_helper(g, i, visited, stack)

    return stack


from typing import TypeVar

T = TypeVar('T')


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


# A possibly faster way to find descendant of a node in a directed ACYCLIC graph
#
# def find_descendants(d, key):
#     """
#     >>> g = dict(a='c', b='ce', c='de', e=['z'], x=['y', 'w'], tt='y')
#     >>> sorted(find_descendants(g, 'a'))
#     ['a', 'c', 'd', 'e', 'z']
#     """
#
#     yield key
#     try:
#         direct_neighbors = d[key]
#         for n in direct_neighbors:
#             yield from find_descendants(d, n)
#     except KeyError:
#         pass
