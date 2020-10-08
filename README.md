# meshed

Tools that enable operations on graphs where grapsh are represented by an adjacency Mapping.

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
```

