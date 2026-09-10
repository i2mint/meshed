# meshed.ext.gk

This module is meant to explore a different representation of a computation graph
and a different way of executing it.
It is based on Yahoo’s graphkit library. The library hasn’t been maintained since 2018,
so vendored and modified here).
One of the main differences is that we got rid of the networkx dependency,
which was used to represent the computation graph.
Instead, this module uses meshed’s itools library to represent the computation graph.

### Yahoo’s graphkit library is under Apache License 2.0:

### Copyright 2016, Yahoo Inc.

### Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

#### NOTE
This module is only meant to an exploratory “extension”. It is not planned to be maintained.

### Functions

| [`get_data_node`](#meshed.ext.gk.get_data_node)(name, graph)           | Gets a data node from a graph using its name                                                              |
|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| [`ready_to_delete_data_node`](#meshed.ext.gk.ready_to_delete_data_node)(name, ...) | Determines if a DataPlaceholderNode is ready to be deleted from the cache.                                |
| [`ready_to_schedule_operation`](#meshed.ext.gk.ready_to_schedule_operation)(op, ...) | Determines if a Operation is ready to be scheduled for execution based on what has already been executed. |

### Classes

| [`Data`](#meshed.ext.gk.Data)(\*\*kwargs)                           | This wraps any data that is consumed or produced by a Operation.                                                                                  |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DataPlaceholderNode`](#meshed.ext.gk.DataPlaceholderNode)                        | A node for the Network graph that describes the name of a Data instance produced or required by a layer.                                          |
| [`DeleteInstruction`](#meshed.ext.gk.DeleteInstruction)                          | An instruction for the compiled list of evaluation steps to free or delete a Data instance from the Network's cache after it is no longer needed. |
| [`FunctionalOperation`](#meshed.ext.gk.FunctionalOperation)(\*\*kwargs)            |                                                                                                                                                   |
| [`Network`](#meshed.ext.gk.Network)(\*\*kwargs)                        | This is the main network implementation.                                                                                                          |
| [`NetworkOperation`](#meshed.ext.gk.NetworkOperation)(\*\*kwargs)               |                                                                                                                                                   |
| [`Operation`](#meshed.ext.gk.Operation)([name, needs, provides, params]) | This is an abstract class representing a data transformation.                                                                                     |
| [`compose`](#meshed.ext.gk.compose)([name, merge])                     | This is a simple class that's used to compose `operation` instances into a computation graph.                                                     |
| [`operation`](#meshed.ext.gk.operation)([fn])                            | This object represents an operation in a computation graph.                                                                                       |
| [`optional`](#meshed.ext.gk.optional)                                   | Input values in `needs` may be designated as optional using this modifier.                                                                        |

### *class* meshed.ext.gk.Data(\*\*kwargs)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

This wraps any data that is consumed or produced
by a Operation. This data should also know how to serialize
itself appropriately.
This class an “abstract” class that should be extended by
any class working with data in the HiC framework.

### *class* meshed.ext.gk.DataPlaceholderNode

Bases: [`str`](https://docs.python.org/3/library/stdtypes.html#str)

A node for the Network graph that describes the name of a Data instance
produced or required by a layer.

### *class* meshed.ext.gk.DeleteInstruction

Bases: [`str`](https://docs.python.org/3/library/stdtypes.html#str)

An instruction for the compiled list of evaluation steps to free or delete
a Data instance from the Network’s cache after it is no longer needed.

### *class* meshed.ext.gk.FunctionalOperation(\*\*kwargs)

Bases: [`Operation`](#meshed.ext.gk.Operation)

### *class* meshed.ext.gk.Network(\*\*kwargs)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

This is the main network implementation. The class contains all of the
code necessary to weave together operations into a directed-acyclic-graph (DAG)
and pass data through.

#### add_op(operation)

Adds the given operation and its data requirements to the network graph
based on the name of the operation, the names of the operation’s needs, and
the names of the data it provides.

* **Parameters:**
  **operation** ([*Operation*](#meshed.ext.gk.Operation)) – Operation object to add.

#### compile()

Create a set of steps for evaluating layers
and freeing memory as necessary

#### compute(outputs, named_inputs, method=None)

Run the graph. Any inputs to the network must be passed in by name.

* **Parameters:**
  * **output** ([*list*](https://docs.python.org/3/library/stdtypes.html#list)) – The names of the data node you’d like to have returned
    once all necessary computations are complete.
    If you set this variable to `None`, all
    data nodes will be kept and returned at runtime.
  * **named_inputs** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dict of key/value pairs where the keys
    represent the data nodes you want to populate,
    and the values are the concrete values you
    want to set for the data node.
* **Returns:**
  a dictionary of output data objects, keyed by name.

#### plot(filename=None, show=False)

Plot the graph.

params:

* **Parameters:**
  * **filename** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Write the output to a png, pdf, or graphviz dot file. The extension
    controls the output format.
  * **show** (*boolean*) – If this is set to True, use matplotlib to show the graph diagram
    (Default: False)
* **Returns:**
  An instance of the pydot graph

#### show_layers()

Shows info (name, needs, and provides) about all layers in this network.

### *class* meshed.ext.gk.NetworkOperation(\*\*kwargs)

Bases: [`Operation`](#meshed.ext.gk.Operation)

#### set_execution_method(method)

Determine how the network will be executed.
:type method: 
:param method: str

> If “parallel”, execute graph operations concurrently
> using a threadpool.

### *class* meshed.ext.gk.Operation(name='None', needs=None, provides=None, params=<factory>)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

This is an abstract class representing a data transformation. To use this,
please inherit from this class and customize the `.compute` method to your
specific application.

Names may be given to this layer and its inputs and outputs. This is
important when connecting layers and data in a Network object, as the
names are used to construct the graph.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The name the operation (e.g. conv1, conv2, etc..)
  * **needs** ([`list`](https://docs.python.org/3/library/stdtypes.html#list)) – Names of input data objects this layer requires.
  * **provides** ([`list`](https://docs.python.org/3/library/stdtypes.html#list)) – Names of output data objects this provides.
  * **params** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict)) – 

    A dict of key/value pairs representing parameters
    associated with your operation. These values will be
    accessible using the `.params` attribute of your object.

    NOTE:
    : It’s important that any values stored in this
      argument must be pickelable.

#### compute(inputs)

This method must be implemented to perform this layer’s feed-forward
computation on a given set of inputs.

* **Parameters:**
  **inputs** ([*list*](https://docs.python.org/3/library/stdtypes.html#list)) – A list of [`Data`](#meshed.ext.gk.Data) objects on which to run the layer’s
  feed-forward computation.
* **Returns list:**
  Should return a list of [`Data`](#meshed.ext.gk.Data) objects representing
  the results of running the feed-forward computation on
  `inputs`.

### *class* meshed.ext.gk.compose(name=None, merge=False)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

This is a simple class that’s used to compose `operation` instances into
a computation graph.

* **Parameters:**
  * **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – A name for the graph being composed by this object.
  * **merge** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If `True`, this compose object will attempt to merge together
    `operation` instances that represent entire computation graphs.
    Specifically, if one of the `operation` instances passed to this
    `compose` object is itself a graph operation created by an
    earlier use of `compose` the sub-operations in that graph are
    compared against other operations passed to this `compose`
    instance (as well as the sub-operations of other graphs passed to
    this `compose` instance).  If any two operations are the same
    (based on name), then that operation is computed only once, instead
    of multiple times (one for each time the operation appears).

### meshed.ext.gk.get_data_node(name, graph)

Gets a data node from a graph using its name

### *class* meshed.ext.gk.operation(fn=None, \*\*kwargs)

Bases: [`Operation`](#meshed.ext.gk.Operation)

This object represents an operation in a computation graph.  Its
relationship to other operations in the graph is specified via its
`needs` and `provides` arguments.

* **Parameters:**
  * **fn** (*function*) – The function used by this operation.  This does not need to be
    specified when the operation object is instantiated and can instead
    be set via `__call__` later.
  * **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The name of the operation in the computation graph.
  * **needs** ([*list*](https://docs.python.org/3/library/stdtypes.html#list)) – Names of input data objects this operation requires.  These should
    correspond to the `args` of `fn`.
  * **provides** ([*list*](https://docs.python.org/3/library/stdtypes.html#list)) – Names of output data objects this operation provides.
  * **params** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – A dict of key/value pairs representing constant parameters
    associated with your operation.  These can correspond to either
    `args` or `kwargs` of 

    ```
    ``
    ```

    fn\`.

### *class* meshed.ext.gk.optional

Bases: [`str`](https://docs.python.org/3/library/stdtypes.html#str)

Input values in `needs` may be designated as optional using this modifier.
If this modifier is applied to an input value, that value will be input to
the `operation` if it is available.  The function underlying the
`operation` should have a parameter with the same name as the input value
in `needs`, and the input value will be passed as a keyword argument if
it is available.

Here is an example of an operation that uses an optional argument:

```default
from graphkit import operation, compose
from graphkit.modifiers import optional

# Function that adds either two or three numbers.
def myadd(a, b, c=0):
    return a + b + c

# Designate c as an optional argument.
graph = compose('mygraph')(
    operator(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
)

# The graph works with and without 'c' provided as input.
assert graph({'a': 5, 'b': 2, 'c': 4})['sum'] == 11
assert graph({'a': 5, 'b': 2})['sum'] == 7
```

### meshed.ext.gk.ready_to_delete_data_node(name, has_executed, graph)

Determines if a DataPlaceholderNode is ready to be deleted from the
cache.

* **Parameters:**
  * **name::** – The name of the data node to check
  * **has_executed** – set
    A set containing all operations that have been executed so far
  * **graph::** – The networkx graph containing the operations and data nodes
* **Returns:**
  A boolean indicating whether the data node can be deleted or not.

### meshed.ext.gk.ready_to_schedule_operation(op, has_executed, graph)

Determines if a Operation is ready to be scheduled for execution based on
what has already been executed.

* **Parameters:**
  * **op::** – The Operation object to check
  * **has_executed** – set
    A set containing all operations that have been executed so far
  * **graph::** – The networkx graph containing the operations and data nodes
* **Returns:**
  A boolean indicating whether the operation may be scheduled for
  execution based on what has already been executed.
