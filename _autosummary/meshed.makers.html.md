# meshed.makers

Makers

This module contains tools to make meshed objects in different ways.

Let’s start with an example where we have some code representing a user story:

```pycon
>>> def user_story():
...     wfs = call(src_to_wf, data_src)
...     chks_iter = map(chunker, wfs)
...     chks = chain(chks_iter)
...     fvs = map(featurizer, chks)
...     model_outputs = map(model, fvs)
```

If the code is compliant (has only function calls and assignments of their result),
we can extract `FuncNode` factories from these lines (uses AST behind the scenes).

```pycon
>>> from meshed.makers import src_to_func_node_factory
>>> fnodes_factories = list(src_to_func_node_factory(user_story))
```

Each factory is a curried version of `FuncNode`, set up to be able to make a `DAG`
equivalent to the user story, once we provide the necessary functions (`call`,
`map`, and `chain`).

```pycon
>>> from functools import partial
>>> assert all(
... isinstance(x, partial) and issubclass(x.func, FuncNode) for x in fnodes_factories
... )
```

See that the `FuncNode` factories are all set up with
`name` (id),
`out` (output variable name),
`bind` (names of the variables where the function will source it’s arguments), and
`func_label` (which can be used when displaying the DAG, or as a key to the function
to use).

```pycon
>>> assert [x.keywords for x in fnodes_factories] == [
...  {'name': 'call',
...   'out': 'wfs',
...   'bind': {0: 'src_to_wf', 1: 'data_src'},
...   'func_label': 'call'},
...  {'name': 'map',
...   'out': 'chks_iter',
...   'bind': {0: 'chunker', 1: 'wfs'},
...   'func_label': 'map'},
...  {'name': 'chain',
...   'out': 'chks',
...   'bind': {0: 'chks_iter'},
...   'func_label': 'chain'},
...  {'name': 'map_04',
...   'out': 'fvs',
...   'bind': {0: 'featurizer', 1: 'chks'},
...   'func_label': 'map'},
...  {'name': 'map_05',
...   'out': 'model_outputs',
...   'bind': {0: 'model', 1: 'fvs'},
...   'func_label': 'map'}
... ]
```

What can we do with that?

Well, provide the functions, so the DAG can actually compute.

You can do it yourself, or get a little help with `mk_fnodes_from_fn_factories`.

```pycon
>>> from meshed.dag import DAG
>>> from meshed.makers import mk_fnodes_from_fn_factories
>>> fnodes = list(mk_fnodes_from_fn_factories(fnodes_factories))
>>> dag = DAG(fnodes)
>>> print(dag.synopsis_string())
src_to_wf,data_src -> call -> wfs
chunker,wfs -> map -> chks_iter
chks_iter -> chain -> chks
featurizer,chks -> map_04 -> fvs
model,fvs -> map_05 -> model_outputs
```

Wait! But we didn’t actually provide the functions we wanted to use!
What happened?!?
What happened is that `mk_fnodes_from_fn_factories` just made some for us.
It used the convenient `meshed.util.mk_place_holder_func` which makes a function
(that happens to actually compute something and be picklable).

```pycon
>>> from inspect import signature
>>> str(signature(dag))
'(src_to_wf, data_src, chunker, featurizer, model)'
```

We can actually call the `dag` and get something meaningful:

```pycon
>>> dag(1, 2, 3, 4, 5)
'map(model=5, fvs=map(featurizer=4, chks=chain(chks_iter=map(chunker=3, wfs=call(src_to_wf=1, data_src=2)))))'
```

If you don’t want `mk_fnodes_from_fn_factories` to do that (because you are in
prod and need to make sure as much as possible is explicitly as expected, you can
simply use a different `factory_to_func` argument. The default one is:

```pycon
>>> from meshed.makers import dlft_factory_to_func
```

which you can also reuse to make your own.
See below how we provide a `name_to_func_map` to specify how `func_label``s should
map to actual functions, and set ``use_place_holder_fallback=False` to make
sure that we don’t ever fallback on a placeholder function as we did above.

```pycon
>>> def _call(x, y):
...     # would use operator.methodcaller('__call__') but doesn't have a __name__
...     return x + y
>>> def _map(x, y):
...     return [x, y]
>>> def _chain(iterable):
...     return sum(iterable)
>>>
>>> factory_to_func = partial(
...     dlft_factory_to_func,
...     name_to_func_map={'map': _map, 'chain': _chain, 'call': _call},
...     use_place_holder_fallback=False
... )
>>>
>>> fnodes = list(mk_fnodes_from_fn_factories(fnodes_factories, factory_to_func))
>>> dag = DAG(fnodes)
```

On the surface, we get the same dag as we had before – at least from the point of view
of the dag signature, names, and relationships between these names:

```pycon
>>> print(dag.synopsis_string())
src_to_wf,data_src -> call -> wfs
chunker,wfs -> map -> chks_iter
chks_iter -> chain -> chks
featurizer,chks -> map_04 -> fvs
model,fvs -> map_05 -> model_outputs
>>> str(signature(dag))
'(src_to_wf, data_src, chunker, featurizer, model)'
```

But see below that the dag is now using the functions we specified:

```pycon
>>> # dag(src_to_wf=1, data_src=2, chunker=3, featurizer=4, model=5)
>>> # will trigger this:
>>> # src_to_wf=1, data_src=2 -> call -> wfs == 1 + 2 == 3
>>> # chunker=3 , wfs=3 -> map -> chks_iter == [3, 3]
>>> # chks_iter=6 -> chain -> chks == 3 + 3 == 6
>>> # featurizer=4, chks=6 -> map_04 -> fvs == [4, 6]
>>> # model=5, fvs=[4, 6] -> map_05 -> model_outputs == [5, [4, 6]]
>>> dag(1, 2, 3, 4, 5)
[5, [4, 6]]
```

### Functions

| `attr_dict`(obj)                                                                                |                                                                                                                                          |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `code_to_dag`([src, func_src, ...])                                                             | Get a `meshed.DAG` from src code                                                                                                         |
| `code_to_digraph`(src)                                                                          |                                                                                                                                          |
| `code_to_fnodes`([src, func_src, ...])                                                          | Get func_nodes from src code                                                                                                             |
| [`dag_to_jdict`](#meshed.makers.dag_to_jdict)(dag, \*[, func_to_jdict])         | Will produce a json-serializable dictionary from a dag.                                                                                  |
| [`dlft_factory_to_func`](#meshed.makers.dlft_factory_to_func)(factory[, ...])           | Get a function for the given factory, using                                                                                              |
| `fnode_to_jdict`(fnode, \*[, func_to_jdict])                                                    |                                                                                                                                          |
| [`func_nodes_to_named_funcs`](#meshed.makers.func_nodes_to_named_funcs)(func_nodes)          |                                                                                                                                          |
| `is_from_ast_module`(o)                                                                         |                                                                                                                                          |
| `iterize`(func)                                                                                 |                                                                                                                                          |
| [`jdict_to_dag`](#meshed.makers.jdict_to_dag)(jdict, \*[, jdict_to_func])       | Will produce a dag from a json-serializable dictionary.                                                                                  |
| `jdict_to_fnode`(jdict, \*[, jdict_to_func])                                                    |                                                                                                                                          |
| `lined_dag`(funcs)                                                                              |                                                                                                                                          |
| [`mk_fnodes_from_fn_factories`](#meshed.makers.mk_fnodes_from_fn_factories)(fnodes_factories)  | Make func nodes from func node factories and a specification of how to make the nodes from these.                                        |
| [`named_funcs_to_func_nodes`](#meshed.makers.named_funcs_to_func_nodes)(named_funcs)         | Make `FuncNode``s from keyword arguments, using the key as the ``.out` of the `FuncNode` and the value as the `.func` of the `FuncNode`. |
| `node_kwargs_to_func_node_factory`(node_kwargs)                                                 |                                                                                                                                          |
| `parse_assignment`(body[, info])                                                                |                                                                                                                                          |
| [`parse_assignment_steps`](#meshed.makers.parse_assignment_steps)(src)                    | Parse source code and generate tuples of information about it.                                                                           |
| `parse_body`(body, \*[, body_index])                                                            |                                                                                                                                          |
| [`parse_steps`](#meshed.makers.parse_steps)(src)                               | Parse source code and generate tuples of information about it.                                                                           |
| [`parsed_to_node_kwargs`](#meshed.makers.parsed_to_node_kwargs)(target_value)            | Extract FuncNode kwargs (name, out, and bind) from ast (target,value) pairs                                                              |
| `robust_ast_parse`(src)                                                                         |                                                                                                                                          |
| [`signed_itemgetter`](#meshed.makers.signed_itemgetter)(\*keys)                      | Like `operator.itemgetter`, except has a signature, which we needed                                                                      |
| `simple_code_to_digraph`(src)                                                                   |                                                                                                                                          |
| [`src_to_func_node_factory`](#meshed.makers.src_to_func_node_factory)(src[, exclude_names]) |                                                                                                                                          |
| [`triples_to_fnodes`](#meshed.makers.triples_to_fnodes)(triples)                     | Converts an iterable of func call triples to an iterable of <br/><br/>```<br/>``<br/>```<br/><br/>FuncNode\`\`s.                         |

### Classes

| [`dlft_factory_to_func_mapping`](#meshed.makers.dlft_factory_to_func_mapping)()   |    |
|-----------------------------------------------------------------------------------|----|

### meshed.makers.dag_to_jdict(dag, , func_to_jdict=None)

Will produce a json-serializable dictionary from a dag.

### meshed.makers.dlft_factory_to_func(factory, name_to_func_map=None, use_place_holder_fallback=True)

Get a function for the given factory, using

### *class* meshed.makers.dlft_factory_to_func_mapping

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

### meshed.makers.extract_tokens(string, pos=0, endpos=9223372036854775807)

Return a list of all non-overlapping matches of pattern in string.

### meshed.makers.func_nodes_to_named_funcs(func_nodes)

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/library/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]

Make some components (kwargs) based on the `.out` and `.func` of the

```
``
```

FuncNode\`\`s.

Example use: To get from `DAG` to `Slabs`.

```pycon
>>> from meshed import DAG, FuncNode
>>> dag = DAG([
...     FuncNode(lambda x: x + 1, out='a'),
...     FuncNode(lambda a: a + 2, out='b',),
...     FuncNode(lambda a, b: a * b, out='c'),
... ])
>>> dag(x=10)
143
>>> named_funcs = func_nodes_to_named_funcs(dag.func_nodes)
>>> isinstance(named_funcs, dict)
True
>>> list(named_funcs)
['a', 'b', 'c']
>>> callable(named_funcs['a'])
True
>>> assert dag.find_func_node('a').func(3) == named_funcs['a'](3) ==4
```

The inverse of this function is `named_funcs_to_func_nodes`.

```pycon
>>> func_nodes = list(named_funcs_to_func_nodes(named_funcs))
>>> dag2 = DAG(func_nodes)
>>> assert dag2(x=3) == dag(x=3) == 24
```

### meshed.makers.jdict_to_dag(jdict, , jdict_to_func=None)

Will produce a dag from a json-serializable dictionary.

### meshed.makers.mk_fnodes_from_fn_factories(fnodes_factories, factory_to_func=<function dlft_factory_to_func>)

Make func nodes from func node factories and a specification of how to make the
nodes from these.

* **Parameters:**
  * **fnodes_factories** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/library/constants.html#Ellipsis), [`FuncNode`](meshed.base.html.md#meshed.base.FuncNode)]]) – An iterable of FuncNodeFactory
  * **factory_to_func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/library/constants.html#Ellipsis), [`FuncNode`](meshed.base.html.md#meshed.base.FuncNode)]], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]) – A function that will give you a function given a
    FuncNodeFactory input (where it will draw the information it needs to know
    what kind of function to make).
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`FuncNode`](meshed.base.html.md#meshed.base.FuncNode)]
* **Returns:**

### meshed.makers.named_funcs_to_func_nodes(named_funcs)

Make `FuncNode``s from keyword arguments, using the key as the ``.out` of the
`FuncNode` and the value as the `.func` of the `FuncNode`.

Example use: To get from `Slabs` to `DAG`.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`FuncNode`](meshed.base.html.md#meshed.base.FuncNode)]

```pycon
>>> from meshed import DAG
>>> func_nodes = list(named_funcs_to_func_nodes(dict(
...     a=lambda x: x + 1,
...     b=lambda a: a + 2,
...     c=lambda a, b: a * b)
... ))
>>> dag = DAG(func_nodes)
>>> dag(x=3)
24
```

The inverse of this function is `func_nodes_to_named_funcs`.

```pycon
>>> named_funcs = func_nodes_to_named_funcs(dag.func_nodes)
>>> dag2 = DAG(named_funcs_to_func_nodes(named_funcs))
>>> assert dag2(x=3) == dag(x=3) == 24
```

### meshed.makers.parse_assignment_steps(src)

Parse source code and generate tuples of information about it.

* **Parameters:**
  **src** – The source string or a python object whose code string can be extracted.
* **Returns:**
  And generator of “target_values”

```pycon
>>> from meshed.makers import parse_steps
>>> def foo():
...     x = func1(a, b=2)
...     y = func2(x, c=3)
>>> target_values = list(parse_steps(foo))
```

Let’s look at the first target_value to see what it contains:

```pycon
>>> name, call = target_values[0]  # a 2-tuple
>>> assert isinstance(name, ast.Name)  # the first element is a ast Name object
>>> sorted(vars(name))
['col_offset', 'ctx', 'end_col_offset', 'end_lineno', 'id', 'lineno']
>>> name.id
'x'
>>> assert isinstance(call, ast.Call)  # the first element is a ast Call object
>>> sorted(vars(call))
['args', 'col_offset', 'end_col_offset', 'end_lineno', 'func', 'keywords', 'lineno']
>>> call.args[0].id
'a'
>>> call.keywords[0].arg
'b'
>>> call.keywords[0].value.value
2
```

Basically, these ast objects contain all we need to know about the (parsed) source.

### meshed.makers.parse_steps(src)

Parse source code and generate tuples of information about it.

* **Parameters:**
  **src** – The source string or a python object whose code string can be extracted.
* **Returns:**
  And generator of “target_values”

```pycon
>>> from meshed.makers import parse_steps
>>> def foo():
...     x = func1(a, b=2)
...     y = func2(x, c=3)
>>> target_values = list(parse_steps(foo))
```

Let’s look at the first target_value to see what it contains:

```pycon
>>> name, call = target_values[0]  # a 2-tuple
>>> assert isinstance(name, ast.Name)  # the first element is a ast Name object
>>> sorted(vars(name))
['col_offset', 'ctx', 'end_col_offset', 'end_lineno', 'id', 'lineno']
>>> name.id
'x'
>>> assert isinstance(call, ast.Call)  # the first element is a ast Call object
>>> sorted(vars(call))
['args', 'col_offset', 'end_col_offset', 'end_lineno', 'func', 'keywords', 'lineno']
>>> call.args[0].id
'a'
>>> call.keywords[0].arg
'b'
>>> call.keywords[0].value.value
2
```

Basically, these ast objects contain all we need to know about the (parsed) source.

### meshed.makers.parsed_to_node_kwargs(target_value)

Extract FuncNode kwargs (name, out, and bind) from ast (target,value) pairs

* **Parameters:**
  **target_value** – A (target, value) pair
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`dict`](https://docs.python.org/3/library/stdtypes.html#dict)]
* **Returns:**
  A `{name:..., out:..., bind:...}` dict (meant to be used to curry FuncNode

Where can you make make target_values? With the `parse_assignment_steps` function.

```pycon
>>> from meshed.makers import parse_assignment_steps
>>> def foo():
...     x = func1(a, b=2)
...     y = func2(x, func1, c=3, d=x)
>>> for target_value in parse_assignment_steps(foo):
...     for d in parsed_to_node_kwargs(target_value):
...         print(d)
{'name': 'func1', 'out': 'x', 'bind': {0: 'a', 'b': 2}}
{'name': 'func2', 'out': 'y', 'bind': {0: 'x', 1: 'func1', 'c': 3, 'd': 'x'}}
```

### meshed.makers.signed_itemgetter(\*keys)

Like `operator.itemgetter`, except has a signature, which we needed

### meshed.makers.src_to_func_node_factory(src, exclude_names=None)

* **Parameters:**
  * **src** – Callable or string of callable.
  * **exclude_names** – Names to exclude when making func_nodes
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`FuncNode`](meshed.base.html.md#meshed.base.FuncNode) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/library/constants.html#Ellipsis), [`FuncNode`](meshed.base.html.md#meshed.base.FuncNode)]]
* **Returns:**

### meshed.makers.triples_to_fnodes(triples)

Converts an iterable of func call triples to an iterable of `FuncNode``s.
(Which in turn can be converted to a ``DAG`.)

Note how the python identifiers are extracted (on the basis of “an unbroken
sequence of alphanumerical (and underscore) characters”, ignoring all other
characters).

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`FuncNode`](meshed.base.html.md#meshed.base.FuncNode)]

```pycon
>>> from meshed import DAG
>>> dag = DAG(
...     triples_to_fnodes(
...     [
...         ('alpha bravo', 'charlie', 'delta echo'),
...         (' foxtrot  &^$#', 'golf', '  alpha,  echo'),
...     ])
... )
>>> print(dag.synopsis_string())
delta,echo -> charlie -> alpha__bravo
alpha__bravo -> alpha__0 -> alpha
alpha__bravo -> bravo__1 -> bravo
alpha,echo -> golf -> foxtrot
```
