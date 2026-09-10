# meshed

`meshed` contains a set of tools that allow the developer to provide a collection
of python objects (think functions) and some policy of how these should be connected
and get an aggregate object that will use the underlying objects in some way.

If you want something concrete, think of the python objects to be functions,
and the aggregation policies to be things like “function composition” (pipelines)
or DAGs.
But the intent is to be able to get more general aggregations than those.

## Extras

`itools.py` contain tools that enable operations on graphs where graphs are represented
by an adjacency Mapping.

### Modules

| [`base`](meshed.base.html.md#module-meshed.base)               | Base functionality of meshed                                                                    |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| [`caching`](meshed.caching.html.md#module-meshed.caching)         | Caching meshes                                                                                  |
| [`components`](meshed.components.html.md#module-meshed.components)   | Specialized components for meshed.                                                              |
| [`composition`](meshed.composition.html.md#module-meshed.composition) | Specific use of FuncNode and DAG                                                                |
| [`dag`](meshed.dag.html.md#module-meshed.dag)                 | Making DAGs                                                                                     |
| [`examples`](meshed.examples.html.md#module-meshed.examples)       | Examples of using meshed.                                                                       |
| [`ext`](meshed.ext.html.md#module-meshed.ext)                 | vendors                                                                                         |
| [`itools`](meshed.itools.html.md#module-meshed.itools)           | Functions that provide iterators of g elements where g is any adjacency Mapping representation. |
| [`makers`](meshed.makers.html.md#module-meshed.makers)           | Makers                                                                                          |
| [`scrap`](meshed.scrap.html.md#module-meshed.scrap)             | For scrap only                                                                                  |
| [`slabs`](meshed.slabs.html.md#module-meshed.slabs)             | Tools to generate slabs.                                                                        |
| [`tools`](meshed.tools.html.md#module-meshed.tools)             | Tools to work with meshed                                                                       |
| [`util`](meshed.util.html.md#module-meshed.util)               | util functions                                                                                  |
| [`viz`](meshed.viz.html.md#module-meshed.viz)                 | Visualization utilities for the meshed package.                                                 |
