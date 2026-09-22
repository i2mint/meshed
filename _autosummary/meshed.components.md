# meshed.components

Ready-made extraction components for meshed graphs.

A `DAG` node needs a `__name__` and a signature to know what var node it reads
and what var node it writes. Plain `operator.itemgetter` and `attrgetter`
objects have no `__name__` and no usable one-parameter signature, so this module
wraps them in `Extractor`, a callable that
carries a chosen name and a single positional-only parameter, and can therefore
be listed directly among the functions given to `DAG`.

Main entry points:

- `Itemgetter`: extracts one item, or a tuple of items, from its input by key.
- `AttrGetter`: extracts one attribute, or a tuple of attributes, from its input.
- `Extractor`: the general form; give it a factory and the parameters to build with.

```pycon
>>> from meshed.components import Itemgetter
>>> get_ab = Itemgetter(['a', 'b'])
>>> get_ab({'a': 1, 'b': 2, 'c': 3})
(1, 2)
```

### Classes

| [`Extractor`](#meshed.components.Extractor)(extractor_factory, extractor_params, \*)   | Callable extracting from its single input, named and signed to be a DAG node.   |
|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|

### *class* meshed.components.Extractor(extractor_factory, extractor_params, , name='extractor', input_name='x')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Callable extracting from its single input, named and signed to be a DAG node.

Calling an instance applies `extractor_factory(extractor_params)` to the input.
The instance carries a chosen `__name__` and a one-parameter signature, which
is what `DAG` needs to wire it to var nodes.

* **Parameters:**
  * **extractor_factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]) – Called once with `extractor_params` to make the
    function that is applied to the input.
  * **extractor_params** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Passed to `extractor_factory`.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Becomes the `__name__` of the instance.
  * **input_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the single positional-only parameter of the instance’s
    signature (the var node a DAG will bind it to).
