# meshed

Tools that enable operations on graphs where graphs are represented by an adjacency Mapping.

Again. 

Graphs: You know them. Networks. 
Nodes and edges, and the ecosystem descriptive or transformative functions surrounding these.
Few languages have builtin support for the graph data structure, but all have their libraries to compensate.

The one you're looking at focuses on the representation of a graph as `Mapping` encoding 
its [adjacency list](https://en.wikipedia.org/wiki/Adjacency_list). 
That is, a dictionary-like interface that specifies the graph by specifying for each node
what nodes it's adjacent to:

```python
assert graph[source_node] == iterator_of_nodes_that_source_node_has_edges_to
```

We emphasize that there is no specific graph instance that you need to squeeze your graph into to
be able to use the functions of `meshed`. Suffices that your graph's structure is expressed by 
that dict-like interface 
-- which grown-ups call `Mapping` (see the `collections.abc` or `typing` standard libs for more information).

You'll find a lot of `Mapping`s around pythons. 
And if the object you want to work with doesn't have that interface, 
you can easily create one using one of the many tools of `py2store` meant exactly for that purpose.


# Examples

```pydocstring
>>> from meshed.itools import edges, nodes, isolated_nodes
>>> graph = dict(a='c', b='ce', c='abde', d='c', e=['c', 'b'], f={})
>>> sorted(edges(graph))
[('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'), ('d', 'c'), ('e', 'b'), ('e', 'c')]
>>> sorted(nodes(graph))
['a', 'b', 'c', 'd', 'e', 'f']
>>> set(isolated_nodes(graph))
{'f'}
>>>
>>> from meshed.makers import edge_reversed_graph
>>> g = dict(a='c', b='cd', c='abd', e='')
>>> assert edge_reversed_graph(g) == {'c': ['a', 'b'], 'd': ['b', 'c'], 'a': ['c'], 'b': ['c'], 'e': []}
>>> reverse_g_with_sets = edge_reversed_graph(g, set, set.add)
>>> assert reverse_g_with_sets == {'c': {'a', 'b'}, 'd': {'b', 'c'}, 'a': {'c'}, 'b': {'c'}, 'e': set([])}
```

