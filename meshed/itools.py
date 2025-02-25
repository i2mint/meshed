"""Functions that provide iterators of g elements where g is any
adjacency Mapping representation.

"""

from typing import (
    Any,
    Mapping,
    Sized,
    MutableMapping,
    Iterable,
    Callable,
    List,
    TypeVar,
    Union,
    Optional,
)
from itertools import product, chain
from functools import wraps, reduce, partial
from collections import defaultdict
from random import sample, randint
from operator import or_

from i2.signatures import Sig

N = TypeVar("N")
Graph = Mapping[N, Iterable[N]]
MutableGraph = MutableMapping[N, Iterable[N]]


def _import_or_raise(module_name, pip_install_name: Optional[Union[str, bool]] = None):
    try:
        return __import__(module_name)
    except ImportError as e:
        if pip_install_name is True:
            pip_install_name = module_name  # use the module name as the install name
        if pip_install_name:
            msg = f"You can install it by running: `pip install {pip_install_name}`"
        else:
            msg = "Please install it first."
        raise ImportError(f"Could not import {module_name}. {msg}") from e


def random_graph(n_nodes: int = 7):
    """Get a random graph.

    >>> random_graph()  # doctest: +SKIP
    {0: [6, 3, 5, 2],
     1: [3, 2, 0, 6],
     2: [5, 6, 4, 0],
     3: [1, 0, 5, 6, 3],
     4: [],
     5: [1, 5, 3, 6],
     6: [4, 3, 1]}
    >>> random_graph(3)  # doctest: +SKIP
    {0: [0], 1: [0], 2: []}
    """
    nodes = range(n_nodes)

    def gen():
        for src in nodes:
            n_dst = randint(0, n_nodes - 1)
            dst = sample(nodes, n_dst)
            yield src, list(dst)

    return dict(gen())


def graphviz_digraph(d: Graph):
    """Makes a graphviz graph using the links specified by dict d"""
    graphviz = _import_or_raise("graphviz", "graphviz")
    dot = graphviz.Digraph()
    for k, v in d.items():
        for vv in v:
            dot.edge(vv, k)
    return dot


def _handle_exclude_nodes(func: Callable):
    sig = Sig(func)

    @wraps(func)
    def _func(*args, **kwargs):
        kwargs = sig.map_arguments(args, kwargs, apply_defaults=True)
        try:
            _exclude_nodes = kwargs["_exclude_nodes"]
        except KeyError:
            raise RuntimeError(f"{func} doesn't have a _exclude_nodes argument")

        if _exclude_nodes is None:
            _exclude_nodes = set()
        elif not isinstance(_exclude_nodes, set):
            _exclude_nodes = set(_exclude_nodes)

        kwargs["_exclude_nodes"] = _exclude_nodes
        args, kwargs = sig.mk_args_and_kwargs(kwargs)
        return func(*args, **kwargs)

    return _func


def add_edge(g: MutableGraph, node1, node2):
    """Add an edge FROM node1 TO node2"""
    if node1 in g:
        g[node1].append(node2)
    else:
        g[node1] = [node2]


def edges(g: Graph):
    """Generates edges of graph, i.e. ``(from_node, to_node)`` tuples.

    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert sorted(edges(g)) == [
    ...     ('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'),
    ...     ('c', 'e'), ('d', 'c'), ('e', 'c'), ('e', 'z')]
    """
    for src in g:
        for dst in g[src]:
            yield src, dst


def nodes(g: Graph):
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


def has_node(g: Graph, node, check_adjacencies=True):
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
def successors(g: Graph, node, _exclude_nodes=None):
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


def predecessors(g: Graph, node):
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


def children(g: Graph, source: Iterable[N]):
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


def parents(g: Graph, source: Iterable[N]):
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
def ancestors(g: Graph, source: Iterable[N], _exclude_nodes=None):
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


def descendants(g: Graph, source: Iterable[N], _exclude_nodes=None):
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
def root_nodes(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(root_nodes(g))
    ['f']

    Note that `f` is present: Isolated nodes are considered both as
    root and leaf nodes both.
    """
    nodes_having_parents = set(chain.from_iterable(g.values()))
    return set(g) - set(nodes_having_parents)


# TODO: Can be made much more efficient, by looking at the ancestors code itself
def root_ancestors(graph: dict, nodes: Union[str, Iterable[str]]):
    """
    Returns the roots of the sub-dag that contribute to compute the given nodes.
    """
    if isinstance(nodes, str):
        nodes = nodes.split()
    get_ancestors = partial(ancestors, graph)
    ancestors_of_nodes = reduce(or_, map(get_ancestors, nodes), set())
    return ancestors_of_nodes & set(root_nodes(graph))


# TODO: Can serious be optimized, and hasn't been tested much: Revise
def leaf_nodes(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(leaf_nodes(g))
    ['f', 'z']

    Note that `f` is present: Isolated nodes are considered both as
    root and leaf nodes both.
    """
    return root_nodes(edge_reversed_graph(g))


def isolated_nodes(g: Graph):
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


def find_path(g: Graph, src, dst, path=None):
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


def reverse_edges(g: Graph):
    """Generator of reversed edges. Like edges but with inverted edges.

    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert sorted(reverse_edges(g)) == [
    ...     ('a', 'c'), ('b', 'c'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'),
    ...     ('d', 'c'), ('e', 'b'), ('e', 'c'), ('z', 'e')]

    NOTE: Not to be confused with  ``edge_reversed_graph`` which inverts the direction
    of edges.
    """
    for src, dst_nodes in g.items():
        yield from product(dst_nodes, src)


def has_cycle(g: Graph) -> List[N]:
    """
        Returns a list representing a cycle in the graph if any. An empty list indicates no cycle.

        :param g: The graph to check for cycles, represented as a dictionary where keys are nodes
                  and values are lists of nodes pointing to the key node (parents of the key node).

        Example usage:
        >>> g = dict(e=['c', 'd'], c=['b'], d=['b'], b=['a'])
        >>> has_cycle(g)
        []

        >>> g['a'] = ['e']  # Introducing a cycle
        >>> has_cycle(g)
        ['e', 'c', 'b', 'a', 'e']

    Design notes:

    - **Graph Representation**: The graph is interpreted such that each key is a child node,
    and the values are lists of its parents. This representation requires traversing
    the graph in reverse, from child to parent, to detect cycles.
    I regret this design choice, which was aligned with the original problem that was
    being solved, but which doesn't follow the usual representation of a graph.
    - **Consistent Return Type**: The function systematically returns a list. A non-empty
    list indicates a cycle (showing the path of the cycle), while an empty list indicates
    the absence of a cycle.
    - **Depth-First Search (DFS)**: The function performs a DFS on the graph to detect
    cycles. It uses a recursion stack (rec_stack) to track the path being explored and
    a visited set (visited) to avoid re-exploring nodes.
    - **Cycle Detection and Path Reconstruction**: When a node currently in the recursion
    stack is encountered again, a cycle is detected. The function then reconstructs the
    cycle path from the current path explored, including the start and end node to
    illustrate the cycle closure.
    - **Efficient Backtracking**: After exploring a node's children, the function
    backtracks by removing the node from the recursion stack and the current path,
    ensuring accurate path tracking for subsequent explorations.

    """
    visited = set()  # Tracks visited nodes to avoid re-processing
    rec_stack = set()  # Tracks nodes currently in the recursion stack to detect cycles

    def _has_cycle(node, path):
        """
        Helper function to perform DFS on the graph and detect cycles.
        :param node: Current node being processed
        :param path: Current path taken from the start node to the current node
        :return: List representing the cycle, empty if no cycle is found
        """
        if node in rec_stack:
            # Cycle detected, return the cycle path including the current node for closure
            cycle_start_index = path.index(node)
            return path[cycle_start_index:] + [node]
        if node in visited:
            # Node already processed and didn't lead to a cycle, skip
            return []

        # Mark the current node as visited and add to the recursion stack
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        # Explore all parent nodes
        for parent in g.get(node, []):
            cycle_path = _has_cycle(parent, path)
            if cycle_path:
                # Cycle found in the path of the parent node
                return cycle_path

        # Current path didn't lead to a cycle, backtrack
        rec_stack.remove(node)
        path.pop()

        return []

    # Iterate over all nodes to ensure disconnected components are also checked
    for node in g:
        cycle_path = _has_cycle(node, [])
        if cycle_path:
            # Return the first cycle found
            return cycle_path

    # No cycle found in any component of the graph
    return []


def out_degrees(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert dict(out_degrees(g)) == (
    ...     {'a': 1, 'b': 2, 'c': 4, 'd': 1, 'e': 2, 'f': 0}
    ... )
    """
    for src, dst_nodes in g.items():
        yield src, len(dst_nodes)


def in_degrees(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert dict(in_degrees(g)) == (
    ... {'a': 1, 'b': 1, 'c': 4,  'd': 1, 'e': 2, 'f': 0, 'z': 1}
    ... )
    """
    return out_degrees(edge_reversed_graph(g))


def copy_of_g_with_some_keys_removed(g: Graph, keys: Iterable):
    keys = _split_if_str(keys)
    return {k: v for k, v in g.items() if k not in keys}


def _topological_sort_helper(g, parent, visited, stack):
    """A recursive function to service topological_sort"""

    visited.add(parent)  # Mark the current node as visited.

    # Recurse for all the vertices adjacent to this node
    for child in reversed(g.get(parent, [])):
        if child not in visited:
            _topological_sort_helper(g, child, visited, stack)

    # Push current node to stack which stores result
    stack.insert(0, parent)
    # print(f"  Inserted {parent}: {stack=}")


def topological_sort(g: Graph):
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
    [0, 4, 2, 3, 1]

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
    for parent in reversed(g):
        if parent not in visited:
            # print(f"Processing {parent}")
            _topological_sort_helper(g, parent, visited, stack)

    return stack


def edge_reversed_graph(
    g: Graph,
    dst_nodes_factory: Callable[[], Iterable[N]] = list,
    dst_nodes_append: Callable[[Iterable[N], N], None] = list.append,
) -> Graph:
    """Invert the from/to direction of the edges of the graph.

    >>> g = dict(a='c', b='cd', c='abd', e='')
    >>> assert edge_reversed_graph(g) == {
    ...     'c': ['a', 'b'], 'd': ['b', 'c'], 'a': ['c'], 'b': ['c'], 'e': []}
    >>> reverse_g_with_sets = edge_reversed_graph(g, set, set.add)
    >>> assert reverse_g_with_sets == {
    ...     'c': {'a', 'b'}, 'd': {'b', 'c'}, 'a': {'c'}, 'b': {'c'}, 'e': set([])}

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


def filter_dict_with_list_values(d, condition):
    return {k: list(filter(condition, d[k])) for k, v in d.items()}


def filter_dict_on_keys(d, condition):
    return {k: v for k, v in d.items() if condition(k, v)}


def nodes_of_graph(graph):
    return set([*graph.keys(), *graph.values()])


def subtract_subgraph(graph, subgraph):
    subnodes = nodes_of_graph(subgraph)
    is_not_subnode = lambda x: x not in subnodes
    not_only_subnodes = lambda x: set(x).issubset(set(subnodes))
    is_not_empty = lambda k, v: len(v) > 0
    key_not_subnode = lambda k, v: k not in subnodes
    graph = filter_dict_with_list_values(graph, is_not_subnode)
    graph = filter_dict_on_keys(graph, is_not_empty)
    graph = filter_dict_on_keys(graph, key_not_subnode)

    return graph
