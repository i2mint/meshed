# meshed.scrap.misc_utils

Misc utils

### Functions

| `coparents_sets`(g, source)                                                                     |                                                           |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `dag_from_funcnodes`(dag, input_names)                                                          |                                                           |
| `extended_family`(g, source)                                                                    |                                                           |
| `funcnode_only`(source)                                                                         |                                                           |
| `kids_of_united_family`(g, source)                                                              |                                                           |
| `known_parents`(g, kid, source)                                                                 |                                                           |
| `list_coparents`(g, coparent)                                                                   |                                                           |
| [`mermaid_pack_nodes`](#meshed.scrap.misc_utils.mermaid_pack_nodes)(mermaid_code, nodes[, ...]) | Output mermaid code with nodes packed into a single node. |

### meshed.scrap.misc_utils.mermaid_pack_nodes(mermaid_code, nodes, packed_node_name=None, , arrow='-->')

Output mermaid code with nodes packed into a single node.

* **Return type:**
  [`str`](https://docs.python.org/3/library/stdtypes.html#str)

```pycon
>>> mermaid_code = '''
... graph TD
...   A --> B
...   B --> C
...   A --> D
...   D --> E
...   E --> C
... '''
>>>
>>>
>>> print(mermaid_pack_nodes(mermaid_code, ['B', 'C', 'E'], 'BCE'))
graph TD
A -->BCE
A --> D
D -->BCE
```
