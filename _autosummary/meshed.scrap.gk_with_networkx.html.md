# meshed.scrap.gk_with_networkx

seriously modified version of yahoo/graphkit

### Classes

| [`Data`](#meshed.scrap.gk_with_networkx.Data)(\*\*kwargs)                           | This wraps any data that is consumed or produced by a Operation.           |
|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`NetworkOperation`](#meshed.scrap.gk_with_networkx.NetworkOperation)(\*\*kwargs)               |                                                                            |
| [`Operation`](#meshed.scrap.gk_with_networkx.Operation)([name, needs, provides, params]) | This is an abstract class representing a data transformation.              |
| [`optional`](#meshed.scrap.gk_with_networkx.optional)                                   | Input values in `needs` may be designated as optional using this modifier. |

### *class* meshed.scrap.gk_with_networkx.Data(\*\*kwargs)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

This wraps any data that is consumed or produced
by a Operation. This data should also know how to serialize
itself appropriately.
This class an “abstract” class that should be extended by
any class working with data in the HiC framework.

### *class* meshed.scrap.gk_with_networkx.NetworkOperation(\*\*kwargs)

Bases: [`Operation`](#meshed.scrap.gk_with_networkx.Operation)

#### set_execution_method(method)

Determine how the network will be executed.
:type method: 
:param method: str

> If “parallel”, execute graph operations concurrently
> using a threadpool.

### *class* meshed.scrap.gk_with_networkx.Operation(name='None', needs=None, provides=None, params=<factory>)

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
  **inputs** ([*list*](https://docs.python.org/3/library/stdtypes.html#list)) – A list of [`Data`](#meshed.scrap.gk_with_networkx.Data) objects on which to run the layer’s
  feed-forward computation.
* **Returns list:**
  Should return a list of [`Data`](#meshed.scrap.gk_with_networkx.Data) objects representing
  the results of running the feed-forward computation on
  `inputs`.

### *class* meshed.scrap.gk_with_networkx.optional

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
