# meshed.itools

Functions that provide iterators of g elements where g is any
adjacency Mapping representation.

### Functions

| [`add_edge`](#meshed.itools.add_edge)(g, node1, node2)                        | Add an edge FROM node1 TO node2                                                            |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`ancestors`](#meshed.itools.ancestors)(g, source[, \_exclude_nodes])          | Set of all nodes (not in source) reachable TO `source` in `g`.                             |
| [`children`](#meshed.itools.children)(g, source)                              | Set of all nodes (not in source) adjacent FROM 'source' in 'g'                             |
| `copy_of_g_with_some_keys_removed`(g, keys)                                                       |                                                                                            |
| [`descendants`](#meshed.itools.descendants)(g, source[, \_exclude_nodes])        | Returns the set of all nodes reachable FROM `source` in `g`.                               |
| [`edge_reversed_graph`](#meshed.itools.edge_reversed_graph)(g[, dst_nodes_factory, ...]) | Invert the from/to direction of the edges of the graph.                                    |
| [`edges`](#meshed.itools.edges)(g)                                         | Generates edges of graph, i.e. `(from_node, to_node)` tuples.                              |
| `filter_dict_on_keys`(d, condition)                                                               |                                                                                            |
| `filter_dict_with_list_values`(d, condition)                                                      |                                                                                            |
| [`find_path`](#meshed.itools.find_path)(g, src, dst[, path])                   | find a path from src to dst nodes in graph                                                 |
| [`graphviz_digraph`](#meshed.itools.graphviz_digraph)(d)                              | Makes a graphviz graph using the links specified by dict d                                 |
| [`has_cycle`](#meshed.itools.has_cycle)(g)                                     | Returns a list representing a cycle in the graph if any. An empty list indicates no cycle. |
| [`has_node`](#meshed.itools.has_node)(g, node[, check_adjacencies])           | Returns True if the graph has given node                                                   |
| [`in_degrees`](#meshed.itools.in_degrees)(g)                                    |                                                                                            |
| [`isolated_nodes`](#meshed.itools.isolated_nodes)(g)                                | Nodes that                                                                                 |
| [`leaf_nodes`](#meshed.itools.leaf_nodes)(g)                                    |                                                                                            |
| [`nodes`](#meshed.itools.nodes)(g)                                         |                                                                                            |
| `nodes_of_graph`(graph)                                                                           |                                                                                            |
| [`out_degrees`](#meshed.itools.out_degrees)(g)                                   |                                                                                            |
| [`parents`](#meshed.itools.parents)(g, source)                               | Set of all nodes (not in source) adjacent TO 'source' in 'g'                               |
| [`predecessors`](#meshed.itools.predecessors)(g, node)                            | Iterator of nodes that have directed paths TO node                                         |
| [`random_graph`](#meshed.itools.random_graph)([n_nodes])                          | Get a random graph.                                                                        |
| [`reverse_edges`](#meshed.itools.reverse_edges)(g)                                 | Generator of reversed edges.                                                               |
| [`root_ancestors`](#meshed.itools.root_ancestors)(graph, nodes)                     | Returns the roots of the sub-dag that contribute to compute the given nodes.               |
| [`root_nodes`](#meshed.itools.root_nodes)(g)                                    |                                                                                            |
| `subtract_subgraph`(graph, subgraph)                                                              |                                                                                            |
| [`successors`](#meshed.itools.successors)(g, node[, \_exclude_nodes])           | Iterator of nodes that have directed paths FROM node                                       |
| [`topological_sort`](#meshed.itools.topological_sort)(g)                              | Return the list of nodes in topological sort order.                                        |

### meshed.itools.add_edge(g, node1, node2)

Add an edge FROM node1 TO node2

### meshed.itools.ancestors(g, source, \_exclude_nodes=None)

Set of all nodes (not in source) reachable TO `source` in `g`.

```pycon
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
```

### meshed.itools.children(g, source)

Set of all nodes (not in source) adjacent FROM ‘source’ in ‘g’

```pycon
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
```

### meshed.itools.descendants(g, source, \_exclude_nodes=None)

Returns the set of all nodes reachable FROM `source` in `g`.

```pycon
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
```

### meshed.itools.edge_reversed_graph(g, dst_nodes_factory=<class 'list'>, dst_nodes_append=<method 'append' of 'list' objects>)

Invert the from/to direction of the edges of the graph.

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`N`), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`N`)]]

```pycon
>>> g = dict(a='c', b='cd', c='abd', e='')
>>> assert edge_reversed_graph(g) == {
...     'c': ['a', 'b'], 'd': ['b', 'c'], 'a': ['c'], 'b': ['c'], 'e': []}
>>> reverse_g_with_sets = edge_reversed_graph(g, set, set.add)
>>> assert reverse_g_with_sets == {
...     'c': {'a', 'b'}, 'd': {'b', 'c'}, 'a': {'c'}, 'b': {'c'}, 'e': set([])}
```

Testing border cases

```pycon
>>> assert edge_reversed_graph(dict(e='', a='e')) == {'e': ['a'], 'a': []}
>>> assert edge_reversed_graph(dict(a='e', e='')) == {'e': ['a'], 'a': []}
```

### meshed.itools.edges(g)

Generates edges of graph, i.e. `(from_node, to_node)` tuples.

```pycon
>>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
>>> assert sorted(edges(g)) == [
...     ('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'),
...     ('c', 'e'), ('d', 'c'), ('e', 'c'), ('e', 'z')]
```

### meshed.itools.find_path(g, src, dst, path=None)

find a path from src to dst nodes in graph

```pycon
>>> g = dict(a='c', b='ce', c=list('abde'), d='c', e=['c', 'z'], f={})
>>> find_path(g, 'a', 'c')
['a', 'c']
>>> find_path(g, 'a', 'b')
['a', 'c', 'b']
>>> find_path(g, 'a', 'z')
['a', 'c', 'b', 'e', 'z']
>>> assert find_path(g, 'a', 'f') == None
```

### meshed.itools.graphviz_digraph(d)

Makes a graphviz graph using the links specified by dict d

### meshed.itools.has_cycle(g)

> Returns a list representing a cycle in the graph if any. An empty list indicates no cycle.
* **Parameters:**
  **g** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`N`), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`N`)]]) – 

  The graph to check for cycles, represented as a dictionary where keys are nodes
  : and values are lists of nodes pointing to the key node (parents of the key node).

  Example usage:
  ```pycon
  >>> g = dict(e=['c', 'd'], c=['b'], d=['b'], b=['a'])
  >>> has_cycle(g)
  []
  ```

  ```pycon
  >>> g['a'] = ['e']  # Introducing a cycle
  >>> has_cycle(g)
  ['e', 'c', 'b', 'a', 'e']
  ```
* **Return type:**
  [`list`](https://docs.python.org/3/library/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`N`)]

Design notes:

- **Graph Representation**: The graph is interpreted such that each key is a child node,
  and the values are lists of its parents. This representation requires traversing
  the graph in reverse, from child to parent, to detect cycles.

I regret this design choice, which was aligned with the original problem that was
being solved, but which doesn’t follow the usual representation of a graph.

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
- **Efficient Backtracking**: After exploring a node’s children, the function
  backtracks by removing the node from the recursion stack and the current path,
  ensuring accurate path tracking for subsequent explorations.

### meshed.itools.has_node(g, node, check_adjacencies=True)

Returns True if the graph has given node

```pycon
>>> g = {
...     0: [1, 2],
...     1: [2]
... }
>>> has_node(g, 0)
True
>>> has_node(g, 2)
True
```

Note that 2 was found, though it’s not a key of `g`.
This shows that we don’t have to have an explicit `{2: []}` in `g`
to be able to see that it’s a node of `g`.
The function will go through the values of the mapping to try to find it
if it hasn’t been found before in the keys.

This can be inefficient, so if that matters, you can express your
graph `g` so that all nodes are explicitly declared as keys, and
use `check_adjacencies=False` to tell the function not to look into
the values of the `g` mapping.

```pycon
>>> has_node(g, 2, check_adjacencies=False)
False
>>> g = {
...     0: [1, 2],
...     1: [2],
...     2: []
... }
>>> has_node(g, 2, check_adjacencies=False)
True
```

### meshed.itools.in_degrees(g)

```pycon
>>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
>>> assert dict(in_degrees(g)) == (
... {'a': 1, 'b': 1, 'c': 4,  'd': 1, 'e': 2, 'f': 0, 'z': 1}
... )
```

### meshed.itools.isolated_nodes(g)

Nodes that

```pycon
>>> g = dict(a='c', b='ce', c=list('abde'), d='c', e=['c', 'z'], f={})
>>> set(isolated_nodes(g))
{'f'}
```

### meshed.itools.leaf_nodes(g)

```pycon
>>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
>>> sorted(leaf_nodes(g))
['f', 'z']
```

Note that `f` is present: Isolated nodes are considered both as
root and leaf nodes both.

### meshed.itools.nodes(g)

```pycon
>>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
>>> sorted(nodes(g))
['a', 'b', 'c', 'd', 'e', 'f', 'z']
```

### meshed.itools.out_degrees(g)

```pycon
>>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
>>> assert dict(out_degrees(g)) == (
...     {'a': 1, 'b': 2, 'c': 4, 'd': 1, 'e': 2, 'f': 0}
... )
```

### meshed.itools.parents(g, source)

Set of all nodes (not in source) adjacent TO ‘source’ in ‘g’

```pycon
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
```

### meshed.itools.predecessors(g, node)

Iterator of nodes that have directed paths TO node

```pycon
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
```

Notice that 2 is a predecessor of 2 here because of the presence
of a 2-1-2 directed path.

### meshed.itools.random_graph(n_nodes=7)

Get a random graph.

```pycon
>>> random_graph()
{0: [6, 3, 5, 2],
 1: [3, 2, 0, 6],
 2: [5, 6, 4, 0],
 3: [1, 0, 5, 6, 3],
 4: [],
 5: [1, 5, 3, 6],
 6: [4, 3, 1]}
>>> random_graph(3)
{0: [0], 1: [0], 2: []}
```

### meshed.itools.reverse_edges(g)

Generator of reversed edges. Like edges but with inverted edges.

```pycon
>>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
>>> assert sorted(reverse_edges(g)) == [
...     ('a', 'c'), ('b', 'c'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'),
...     ('d', 'c'), ('e', 'b'), ('e', 'c'), ('z', 'e')]
```

#### NOTE
Not to be confused with  `edge_reversed_graph` which inverts the direction
of edges.

### meshed.itools.root_ancestors(graph, nodes)

Returns the roots of the sub-dag that contribute to compute the given nodes.

### meshed.itools.root_nodes(g)

```pycon
>>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
>>> sorted(root_nodes(g))
['f']
```

Note that `f` is present: Isolated nodes are considered both as
root and leaf nodes both.

### meshed.itools.successors(g, node, \_exclude_nodes=None)

Iterator of nodes that have directed paths FROM node

```pycon
>>> g = {
...     0: [1, 2],
...     1: [2, 3, 4],
...     2: [1, 4],
...     3: [4]}
>>> assert set(successors(g, 1)) == {1, 2, 3, 4}
>>> assert set(successors(g, 3)) == {4}
>>> assert set(successors(g, 4)) == set()
```

Notice that 1 is a successor of 1 here because there’s a 1-2-1 directed path

### meshed.itools.topological_sort(g)

Return the list of nodes in topological sort order.

This order is such that a node parents will all occur before;
: If order[i] is parent of order[j] then i < j

This is often used to compute the order of computation in a DAG.

```pycon
>>> g = {
...     0: [4, 2],
...     4: [3, 1],
...     2: [3],
...     3: [1]
... }
>>>
>>> list(topological_sort(g))
[0, 4, 2, 3, 1]
```

Here’s an ascii art of the graph, to verify that the topological sort is
indeed as expected.

```default
```

┌───┐     ┌───┐     ┌───┐     ┌───┐
│ 0 │ ──▶ │ 2 │ ──▶ │ 3 │ ──▶ │ 1 │
└───┘     └───┘     └───┘     └───┘

> │                   ▲         ▲
> │                   │         │
> ▼                   │         │

┌───┐                 │         │
│ 4 │ ────────────────┼─────────┘
└───┘                 │

> │                   │
> └───────────────────┘
