# meshed.dag

Making DAGs

Main entry points:

- `DAG`: a callable graph of functions; give it functions or `FuncNode` objects,
  call it with the root variables and get the leaf outputs back.
- `FuncNode` (`meshed.base`): wraps one function with the node `name`, the
  variables its parameters `bind` to, and the `out` variable it writes.
- `ch_funcs`: copy of a DAG with some of its node functions replaced.
- `ch_names`: copy of a DAG with its variables and function nodes renamed.
- `code_to_dag` (`meshed.makers`): build a DAG from Python source (a function
  or a string) whose statements are `out = func(...)` assignments.

In it’s simplest form, consider this:

```pycon
>>> from meshed import DAG
>>>
>>> def this(a, b=1):
...     return a + b
...
>>> def that(x, b=1):
...     return x * b
...
>>> def combine(this, that):
...     return (this, that)
...
>>>
>>> dag = DAG((this, that, combine))
>>> print(dag.synopsis_string())
a,b -> this_ -> this
x,b -> that_ -> that
this,that -> combine_ -> combine
```

But don’t be fooled: There’s much more to it!

## FAQ and Troubleshooting

### DAGs and Pipelines

```pycon
>>> from functools import partial
>>> from meshed import DAG
>>> def chunker(sequence, chk_size: int):
...     return zip(*[iter(sequence)] * chk_size)
>>>
>>> my_chunker = partial(chunker, chk_size=3)
>>> def to_list(iterable):
...     return list(iterable)
>>>
>>> vec = range(8)  # when appropriate, use easier to read sequences
>>> to_list(my_chunker(vec))
[(0, 1, 2), (3, 4, 5)]
```

Oh, that’s just a `my_chunker -> to_list` pipeline!
A pipeline is a subset of DAG, so let me do this:

```pycon
>>> dag = DAG([my_chunker, to_list])
>>> dag(vec)
Traceback (most recent call last):
...
TypeError: missing a required argument: 'iterable'
```

What happened here?
You’re assuming that saying `[my_chunker, to_list]` is enough for DAG to know that
what you meant is for `my_chunker` to feed it’s input to `to_list`.
Sure, DAG has enough information to do so, but the default connection policy doesn’t
assume that it’s a pipeline you want to make.
In fact, the order you specify the functions doesn’t have an affect on the connections
with the default connection policy.

See what the signature of `dag` is:

```pycon
>>> from inspect import signature
>>> str(signature(dag))
'(sequence, iterable, *, chk_size: int = 3)'
```

So dag actually works just fine. Here’s the proof:

```pycon
>>> chunks, as_list = dag(vec, [1, 2, 3])
>>> list(chunks), as_list
([(0, 1, 2), (3, 4, 5)], [1, 2, 3])
```

It’s just not what you might have intended.

Your best bet to get what you intended is to be explicit.

The way to be explicit is to not specify functions alone, but `FuncNodes` that
wrap them, along with the specification
the `name` the function will be referred to by,
the names that it’s parameters should `bind` to (that is, where the function
will get it’s import arguments from), and
the `out` name of where it should be it’s output.

In the current case a fully specified DAG would look something like this:

```pycon
>>> from meshed import FuncNode
>>> dag = DAG(
...     [
...         FuncNode(
...             func=my_chunker,
...             name='chunker',
...             bind=dict(sequence='sequence', chk_size='chk_size'),
...             out='chks'
...         ),
...         FuncNode(
...             func=to_list,
...             name='gather_chks_into_list',
...             bind=dict(iterable='chks'),
...             out='list_of_chks'
...         ),
...     ]
... )
>>> list(dag(vec))
[(0, 1, 2), (3, 4, 5)]
```

But really, if you didn’t care about the names of things,
all you need in this case was to make sure that the output of `my_chunker` was
fed to `to_list`, and therefore the following was sufficient:

```pycon
>>> dag = DAG([
...     FuncNode(my_chunker, out='chks'),  # call the output of chunker "chks"
...     FuncNode(to_list, bind=dict(iterable='chks'))  # source to_list input from "chks"
... ])
>>> list(dag(vec))
[(0, 1, 2), (3, 4, 5)]
```

Connection policies are very useful when you want to define ways for DAG to
“just figure it out” for you.
That is, you want to tell the machine to adapt to your thoughts, not vice versa.
We support such technological expectations!
The default connection policy is there to provide one such ways, but
by all means, use another!

Does this mean that connection policies are not for production code?
Well, it depends. The Zen of Python (`import this`)
states “explicit is better than implicit”, and indeed it’s often
a good fallback rule.
But defining components and the way they should be assembled can go a long way
in achieving consistency, separation of concerns, adaptability, and flexibility.
All quite useful things. Also in production. Especially in production.
That said it is your responsiblity to use the right policy for your particular context.

### Functions

| [`arg_names`](#meshed.dag.arg_names)(func, func_name[, exclude_names])      | List the parameter names of `func`, replacing any found in `exclude_names` with a free `<func_name>__<name>` variant.                                                                                          |
|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`attribute_vals`](#meshed.dag.attribute_vals)(objs, attrs[, egress])            | Extract attributes from an iterable of objects                                                                                                                                                                 |
| [`call_func`](#meshed.dag.call_func)(func, kwargs)                          | Re-key `kwargs` by each key's `__name__` and pass the resulting dict to `Sig(func).source_kwargs`.                                                                                                             |
| `ch_funcs`([func_nodes, func_mapping, ...])                                                       | Copy a DAG (or iterable of func nodes) with some of its node functions replaced.                                                                                                                               |
| `ch_names`([func_nodes, renamer])                                                                 | Renames variables and functions of a `DAG` or iterable of `FuncNodes`.                                                                                                                                         |
| `change_funcs`([func_nodes, func_mapping, ...])                                                   | Copy a DAG (or iterable of func nodes) with some of its node functions replaced.                                                                                                                               |
| [`change_value_on_cond`](#meshed.dag.change_value_on_cond)(d, cond, func)              | Replace, in place, each value `v` of `d` where `cond(k, v)` holds with `func(v)`, and return `d`.                                                                                                              |
| [`dag_to_code`](#meshed.dag.dag_to_code)(dag)                                 | Convert a DAG to code.                                                                                                                                                                                         |
| [`dflt_debugger_feedback`](#meshed.dag.dflt_debugger_feedback)(func_node, scope, ...)    | Print the step number, func node and scope, then return `output` unchanged (default feedback of `DAG.debugger`).                                                                                               |
| [`find_first_free_name`](#meshed.dag.find_first_free_name)(prefix[, ...])              | Return `prefix` if not in `exclude_names`, else the first free `prefix__<i>` with `i` counting up from `start_at`.                                                                                             |
| [`funcnodes_from_pairs`](#meshed.dag.funcnodes_from_pairs)(pairs)                      | Make one mock func node per `(arg, out)` pair (see `mk_mock_funcnode`).                                                                                                                                        |
| [`hook_up`](#meshed.dag.hook_up)(func, variables[, output_name])          | Source inputs and write outputs to given variables mapping.                                                                                                                                                    |
| [`mk_func_name`](#meshed.dag.mk_func_name)(func[, exclude_names])              | Derive a name for `func` (its `__name__`, a generated lambda name, or the wrapped function's name for a `partial`) that is not in `exclude_names`.                                                             |
| [`mk_list_names_unique`](#meshed.dag.mk_list_names_unique)(nodes[, exclude_names])     | List the `.name` of each node, suffixing repeats (and names in `exclude_names`) with `__<i>` so all are distinct.                                                                                              |
| [`mk_mock_funcnode`](#meshed.dag.mk_mock_funcnode)(arg, out)                       | Make a `FuncNode` whose no-op function takes the single parameter `arg` and writes to `out`, named `_mock_<arg>_<out>`.                                                                                        |
| [`mk_nodes_names_unique`](#meshed.dag.mk_nodes_names_unique)(nodes)                     | Set each node's `.name` in place to the unique names of `mk_list_names_unique` and return `nodes`.                                                                                                             |
| [`modified_func_node`](#meshed.dag.modified_func_node)(func_node, \*\*modifications) | Make a new `FuncNode` from `func_node` with some of `func`, `name`, `bind` and `out` replaced by `modifications`.                                                                                              |
| [`named_partial`](#meshed.dag.named_partial)(func, \*args[, \_\_name_\_])       | functools.partial, but with a \_\_name_\_                                                                                                                                                                      |
| [`order_subset_from_list`](#meshed.dag.order_subset_from_list)(items, sublist)           | Sort `sublist` by the position its elements have in `items`.                                                                                                                                                   |
| [`parametrized_dag_factory`](#meshed.dag.parametrized_dag_factory)(dag, param_var_nodes)   | Constructs a factory for sub-DAGs derived from the input DAG, with values of specific 'parameter' variable nodes precomputed and fixed.                                                                        |
| [`partialized_funcnodes`](#meshed.dag.partialized_funcnodes)(func_nodes, ...)           | Yield the func nodes, replacing the function of any node whose parameters include a `keyword_defaults` name with a partial where those parameters are defaulted and moved last; other nodes are yielded as is. |
| [`print_dag_string`](#meshed.dag.print_dag_string)(dag[, bind_info])               | Print `dag.synopsis_string(bind_info)`; the default shows an input as `param=var` only where the two names differ.                                                                                             |
| `rename_nodes`([func_nodes, renamer])                                                             | Renames variables and functions of a `DAG` or iterable of `FuncNodes`.                                                                                                                                         |
| [`reorder_on_constraints`](#meshed.dag.reorder_on_constraints)(funcnodes, outs)          | Topologically sort `funcnodes` after appending (in place) mock nodes that chain each `outs` element to the next, print the order, and return `(func_nodes, var_nodes)` without the mock nodes.                 |

### Classes

| [`DAG`](#meshed.dag.DAG)([func_nodes, cache_last_scope, ...])   | A callable graph of functions: root variables in, leaf variables out.   |
|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|

### *class* meshed.dag.DAG(func_nodes=(), cache_last_scope=True, parameter_merge=functools.partial(<function parameter_merger>, same_kind=True, same_default=True, same_annotation=True), new_scope=<class 'dict'>, name=None, extract_output_from_scope=<function extract_values>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A callable graph of functions: root variables in, leaf variables out.

```pycon
>>> from meshed.dag import DAG, Sig
>>>
>>> def this(a, b=1):
...     return a + b
>>> def that(x, b=1):
...     return x * b
>>> def combine(this, that):
...     return (this, that)
>>>
>>> dag = DAG((this, that, combine))
>>> print(dag.synopsis_string())
a,b -> this_ -> this
x,b -> that_ -> that
this,that -> combine_ -> combine
```

But what does it do?

It’s a callable, with a signature:

```pycon
>>> Sig(dag)
<Sig (a, x, b=1)>
```

And when you call it, it executes the dag from the root values you give it and
returns the leaf output values.

```pycon
>>> dag(1, 2, 3)  # (a+b,x*b) == (1+3,2*3) == (4, 6)
(4, 6)
>>> dag(1, 2)  # (a+b,x*b) == (1+1,2*1) == (2, 2)
(2, 2)
```

The above DAG was created straight from the functions, using only the names of the
functions and their arguments to define how to hook the network up.

But if you didn’t write those functions specifically for that purpose, or you want
to use someone else’s functions, we got you covered.

You can define the name of the node (the `name` argument), the name of the output
(the `out` argument) and a mapping from the function’s arguments names to
“network names” (through the `bind` argument).
The edges of the DAG are defined by matching `out` TO `bind`.

#### add_edge(from_node, to_node, to_param=None)

Add an e

* **Parameters:**
  * **from_node**
  * **to_node**
  * **to_param**
* **Returns:**
  A new DAG with the edge added

```pycon
>>> def f(a, b): return a + b
>>> def g(c, d=1): return c * d
>>> def h(x, y=1): return x ** y
>>>
>>> three_funcs = DAG([f, g, h])
>>> assert (
...     three_funcs(x=1, c=2, a=3, b=4)
...     == (7, 2, 1)
...     == (f(a=3, b=4), g(c=2), h(x=1))
...     == (3 + 4, 2*1, 1** 1)
... )
>>> print(three_funcs.synopsis_string())
a,b -> f_ -> f
c,d -> g_ -> g
x,y -> h_ -> h
>>> hg = three_funcs.add_edge('h', 'g')
>>> assert (
...     hg(a=3, b=4, x=1)
...     == (7, 1)
...     == (f(a=3, b=4), g(c=h(x=1)))
...     == (3 + 4, 1 * (1 ** 1))
... )
>>> print(hg.synopsis_string())
a,b -> f_ -> f
x,y -> h_ -> h
h,d -> g_ -> g
>>>
>>> fhg = three_funcs.add_edge('h', 'g').add_edge('f', 'h')
>>> assert (
...     fhg(a=3, b=4)
...     == 7
...     == g(h(f(3, 4)))
...     == ((3 + 4) * 1) ** 1
... )
>>> print(fhg.synopsis_string())
a,b -> f_ -> f
f,y -> h_ -> h
h,d -> g_ -> g
```

The from and to nodes can be expressed by the `FuncNode` `name` (identifier)
or `out`, or even the function itself if it’s used only once in the `DAG`.

```pycon
>>> fhg = three_funcs.add_edge(h, 'g').add_edge('f_', 'h')
>>> assert fhg(a=3, b=4) == 7
```

By default, the edge will be added from `from_node.out` to the first
parameter of the function of `to_node`.
But if you want otherwise, you can specify the parameter the edge should be
connected to.
For example, see below how we connect the outputs of `g` and `h` to the
parameters `a` and `b` of `f` respectively:

```pycon
>>> f_of_g_and_h = (
...     DAG([f, g, h])
...     .add_edge(g, f, to_param='a')
...     .add_edge(h, f, 'b')
... )
>>> assert (
...     f_of_g_and_h(x=2, c=3, y=2, d=2)
...     == 10
...     == f(g(c=3, d=2), h(x=2, y=2))
...     == 3 * 2 + 2 ** 2
... )
>>>
>>> print(f_of_g_and_h.synopsis_string())
c,d -> g_ -> g
x,y -> h_ -> h
g,h -> f_ -> f
```

See Also `DAG.add_edges` to add multiple edges at once

#### add_edges(edges)

Adds multiple edges by applying `DAG.add_edge` multiple times.

* **Parameters:**
  **edges** – An iterable of `(from_node, to_node)` pairs or
  `(from_node, to_node, param)` triples.
* **Returns:**
  A new dag with the said edges added.

```pycon
>>> def f(a, b): return a + b
>>> def g(c, d=1): return c * d
>>> def h(x, y=1): return x ** y
>>> fhg = DAG([f, g, h]).add_edges([(h, 'g'), ('f_', 'h')])
>>> assert fhg(a=3, b=4) == 7
```

#### bindings_cleaner()

Make func node names unique and rewrite bind values that name a func node into
that node’s `out` (called at the end of `__post_init__`).

#### call_on_scope(scope=None)

Calls the func_nodes using scope (a dict or MutableMapping) both to
source it’s arguments and write it’s results.

#### NOTE
This method is only meant to be used as a backend to \_\_call_\_, not as
an actual interface method. Additional control/constraints on read and writes
can be implemented by providing a custom scope for that. For example, one could
log read and/or writes to specific keys, or disallow overwriting to an existing
key (useful for pipeline sanity), etc.

#### call_on_scope_iteratively(scope=None)

Calls the `func_nodes` using scope (a dict or MutableMapping) both to
source it’s arguments and write it’s results.

Use this function to control each func_node call step iteratively
(through a generator)

#### ch_funcs(ch_func_node_func=<function ch_func_node_func>, /, \*\*func_mapping)

Change some of the functions in the DAG.
More preciseluy get a copy of the DAG where in some of the functions have
changed.

* **Parameters:**
  **name_and_func** – `name=func` pairs where `name` is the
  `FuncNode.name` of the func nodes you want to change and func is the
  function you want to change it by.
* **Return type:**
  [`DAG`](#meshed.dag.DAG)
* **Returns:**
  A new DAG with the different functions.

```pycon
>>> from meshed import FuncNode, DAG
>>> from i2 import Sig
>>>
>>> def f(a, b):
...     return a + b
...
>>>
>>> def g(a_plus_b, x):
...     return a_plus_b * x
...
>>> f_node = FuncNode(func=f, out='a_plus_b')
>>> g_node = FuncNode(func=g, bind={'x': 'b'})
>>> d = DAG((f_node, g_node))
>>> print(d.synopsis_string())
a,b -> f -> a_plus_b
b,a_plus_b -> g_ -> g
>>> d(2, 3)  # (2 + 3) * 3 == 5 * 3
15
>>> dd = d.ch_funcs(f=lambda a, b: a - b)
>>> dd(2, 3)  # (2 - 3) * 3 == -1 * 3
-3
```

You can reference the `FuncNode` you want to change through its `.name` or
`.out` attribute (both are unique to this `FuncNode` in a `DAG`).

```pycon
>>> from i2 import Sig
>>>
>>> dag = DAG([
...     FuncNode(lambda a, b: a + b, name='f'),
...     FuncNode(lambda y=1, z=2: y * z, name='g', bind={'z': 'f'})
... ])
>>>
>>> Sig(dag)
<Sig (a, b, f=2, y=1)>
>>>
>>> dag.ch_funcs(g=lambda y=1, z=2: y / z)
DAG(func_nodes=[FuncNode(a,b -> f -> _f), FuncNode(z=_f,y -> g -> _g)], name=None)
```

But if you change the signature, even slightly you get an error.

Here we didn’t include the defaults:

```pycon
>>> dag.ch_funcs(g=lambda y, z: y / z)
Traceback (most recent call last):
  ...
ValueError: You can only change the func of a FuncNode with a another func if the signatures match.
  ...
```

Here we include defaults, but `z`’s is different:

```pycon
>>> dag.ch_funcs(g=lambda y=1, z=200: y / z)
Traceback (most recent call last):
  ...
ValueError: You can only change the func of a FuncNode with a another func if the signatures match.
  ...
```

Here the defaults are exactly the same, but the order of parameters is
different:

```pycon
>>> dag.ch_funcs(g=lambda z=2, y=1: y / z)
Traceback (most recent call last):
  ...
ValueError: You can only change the func of a FuncNode with a another func if the signatures match.
  ...
```

This validation of the functions controlled by the `func_comparator`
argument. By default this is the `compare_signatures` which compares the
signatures of the functions in the strictest way possible.
The is the right choice for a default since it will get you out of trouble
down the line.

But it’s also annoying in many situations, and in those cases you should
specify the `func_comparator` that makes sense for your context.

Since most of the time, you’ll want to compare functions solely based on
their signature, we provide a `compare_signatures` allows you to control the
signature comparison through a `signature_comparator` argument.

```pycon
>>> from meshed import compare_signatures
>>> from functools import partial
>>> on_names = lambda sig1, sig2: list(sig1.parameters) == list(sig2.parameters)
>>> same_names = partial(compare_signatures, signature_comparator=on_names)
>>> ch_fnode = partial(ch_func_node_func, func_comparator=same_names)
>>> d = dag.ch_funcs(ch_fnode, g=lambda y, z: y / z);
>>> Sig(d)
<Sig (a, b, y)>
>>> d(2, 3, 4)
0.8
```

And this one works too:

```pycon
>>> d = dag.ch_funcs(ch_fnode, g=lambda y=1, z=200: y / z);
```

But our `same_names` function compared names including their order.
If we want a function with the signature `(z=2, y=1)` to be able to be
“injected” we’ll need a different comparator:

```pycon
>>> _names = lambda sig1, sig2: set(sig1.parameters) == set(sig2.parameters)
>>> same_set_of_names = partial(
...     compare_signatures,
...     signature_comparator=(
...         lambda sig1, sig2: set(sig1.parameters) == set(sig2.parameters)
...     )
... )
>>> ch_fnode2 = partial(ch_func_node_func, func_comparator=same_set_of_names)
>>> d = dag.ch_funcs(ch_fnode2, g=lambda z=2, y=1: y / z);
```

#### copy(renamer=<function numbered_suffix_renamer>)

Make a new `DAG` from renamed copies of the func nodes (see `ch_names` for what `renamer` may be).

With the default renamer every variable and function node gets a `_1`
suffix (or an incremented one):

```pycon
>>> def f(a, b):
...     return a + b
>>> def g(f, c):
...     return f * c
>>> dag = DAG([f, g])
>>> print(dag.copy().synopsis_string())
a_1,b_1 -> f__1 -> f_1
f_1,c_1 -> g__1 -> g_1
```

#### debugger(feedback=<function dflt_debugger_feedback>)

Utility to debug DAGs by computing each step sequentially, with feedback.

* **Parameters:**
  **feedback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A callable that defines what feedback is given, usually used to
  print/log some information and output some information for every step.
  Must be a function with signature `(func_node, scope, output, step)` or
  a subset thereof.
* **Returns:**

```pycon
>>> from inspect import signature
>>>
>>> def f(a, b):
...     return a + b
...
>>> def g(c, d=4):
...     return c * d
...
>>> def h(f, g):
...     return g - f
...
>>> dag2 = DAG([f, g, h], name='arithmetic')
>>> dag2
DAG(func_nodes=[FuncNode(a,b -> f_ -> f), FuncNode(c,d -> g_ -> g), FuncNode(f,g -> h_ -> h)], name='arithmetic')
>>> str(signature(dag2))
'(a, b, c, d=4)'
>>> dag2(1,2,3)
9
>>>
>>> debugger = dag2.debugger()
>>> str(signature(debugger))
'(a, b, c, d=4)'
>>> d = debugger(1,2,3)
>>> next(d)
0 --------------------------------------------------------------
    func_node=FuncNode(a,b -> f_ -> f)
    scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3}
3
>>> next(d)
1 --------------------------------------------------------------
    func_node=FuncNode(c,d -> g_ -> g)
    scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3, 'g': 12}
12
```

… and so on. You can also choose to run every step all at once, collecting
the `feedback` outputs of each step in a list, like this:

```pycon
>>> feedback_outputs = list(debugger(1,2,3))
0 --------------------------------------------------------------
    func_node=FuncNode(a,b -> f_ -> f)
    scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3}
1 --------------------------------------------------------------
    func_node=FuncNode(c,d -> g_ -> g)
    scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3, 'g': 12}
2 --------------------------------------------------------------
    func_node=FuncNode(f,g -> h_ -> h)
    scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3, 'g': 12, 'h': 9}
```

#### dot_digraph(start_lines=(), , end_lines=(), vnode_shape='none', fnode_shape='box', func_display=True)

Make lines for dot (graphviz) specification of DAG

```pycon
>>> def add(a, b=1): return a + b
>>> def mult(x, y=3): return x * y
>>> def exp(mult, a): return mult ** a
>>> func_nodes = [
...     FuncNode(add, out='x'), FuncNode(mult, name='the_product'), FuncNode(exp)
... ]
>>> lines = list(DAG(func_nodes).dot_digraph_body())
>>> lines[0]
'x [label="x" shape="none"]'
```

#### dot_digraph_ascii(start_lines=(), , end_lines=(), vnode_shape='none', fnode_shape='box', func_display=True)

Make lines for dot (graphviz) specification of DAG

```pycon
>>> def add(a, b=1): return a + b
>>> def mult(x, y=3): return x * y
>>> def exp(mult, a): return mult ** a
>>> func_nodes = [
...     FuncNode(add, out='x'), FuncNode(mult, name='the_product'), FuncNode(exp)
... ]
>>> lines = list(DAG(func_nodes).dot_digraph_body())
>>> lines[0]
'x [label="x" shape="none"]'
```

#### dot_digraph_body(start_lines=(), , end_lines=(), vnode_shape='none', fnode_shape='box', func_display=True)

Make lines for dot (graphviz) specification of DAG

```pycon
>>> def add(a, b=1): return a + b
>>> def mult(x, y=3): return x * y
>>> def exp(mult, a): return mult ** a
>>> func_nodes = [
...     FuncNode(add, out='x'), FuncNode(mult, name='the_product'), FuncNode(exp)
... ]
>>> lines = list(DAG(func_nodes).dot_digraph_body())
>>> lines[0]
'x [label="x" shape="none"]'
```

#### extract_output_from_scope(keys)

Extract values from dict `d`, returning them:

- as a tuple if len(keys) > 1
- a single value if len(keys) == 1
- None if not

This is used as the default extractor in DAG

```pycon
>>> extract_values({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
(1, 3)
```

Order matters!

```pycon
>>> extract_values({'a': 1, 'b': 2, 'c': 3}, ['c', 'a'])
(3, 1)
```

#### find_func_node(node, default=None)

Return the `FuncNode` that `node` refers to, or `default` when nothing matches.

A `FuncNode` is returned as is; anything else is looked up as a node name,
an `out`, or a function unique in the DAG.

```pycon
>>> def f(a, b):
...     return a + b
>>> dag = DAG([f])
>>> dag.find_func_node('f')  # by out
FuncNode(a,b -> f_ -> f)
>>> dag.find_func_node('f_')  # by name
FuncNode(a,b -> f_ -> f)
>>> dag.find_func_node(f)  # by function
FuncNode(a,b -> f_ -> f)
>>> dag.find_func_node('nope') is None
True
```

#### find_funcs(filt=None)

Yield the `.func` of the func nodes for which `filt` is true (all of them
when `filt` is `None`).

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]

#### *classmethod* from_funcs(\*funcs, \*\*named_funcs)

* **Parameters:**
  * **funcs**
  * **named_funcs**
* **Returns:**

```pycon
>>> dag = DAG.from_funcs(
...     lambda a: a * 2,
...     x=lambda: 10,
...     y=lambda x, _0: x + _0  # _0 refers to first arg (lambda a: a * 2)
... )
>>> print(dag.synopsis_string())
a -> _0_ -> _0
 -> x_ -> x
x,_0 -> y_ -> y
>>> dag(3)
16
```

#### get_node_matching(idx)

Return `idx` itself if it names a var node, else the `FuncNode` that `idx` (a node name, an `out`, or a function unique in the DAG) indexes.

A string matching no node raises `KeyError`; an `idx` that is neither a
string nor a callable raises `NotFound`.

#### *property* graph_ids

The dict representing the `{from_node: to_nodes}` graph.
Like `.graph`, but with node ids (names).

```pycon
>>> from meshed.dag import DAG
>>> def add(a, b=1): return a + b
>>> def mult(x, y=3): return x * y
>>> def exp(mult, a): return mult ** a
>>> assert DAG([add, mult, exp]).graph_ids == {
...     'a': ['add_', 'exp_'],
...     'b': ['add_'],
...     'add_': ['add'],
...     'x': ['mult_'],
...     'y': ['mult_'],
...     'mult_': ['mult'],
...     'mult': ['exp_'],
...     'exp_': ['exp']
... }
```

#### new_scope

alias of [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### parameter_merge(, same_name=True, same_kind=True, same_default=True, same_annotation=True)

Validates that all the params are exactly the same, returning the first if so.

This is used when hooking up functions that use the same parameters (i.e. arg
names). When the name of an argument is used more than once, which kind, default,
and annotation should be used in the interface of the DAG?

If they’re all the same, there’s no problem.

But if they’re not the same, we need to provide control on which to ignore.

```pycon
>>> from inspect import Parameter as P
>>> PK = P.POSITIONAL_OR_KEYWORD
>>> KO = P.KEYWORD_ONLY
>>> parameter_merger(P('a', PK), P('a', PK))
<Parameter "a">
>>> parameter_merger(P('a', PK), P('different_name', PK), same_name=False)
<Parameter "a">
>>> parameter_merger(P('a', PK), P('a', KO), same_kind=False)
<Parameter "a">
>>> parameter_merger(P('a', PK), P('a', PK,  default=42), same_default=False)
<Parameter "a">
>>> parameter_merger(P('a', PK, default=42), P('a', PK), same_default=False)
<Parameter "a=42">
>>> parameter_merger(P('a', PK, annotation=int), P('a', PK), same_annotation=False)
<Parameter "a: int">
```

#### *property* params_for_src

The `{src_name: list_of_params_using_that_src,...}` dictionary.
That is, a `dict` having lists of all `Parameter` objs that are used by a
`node.bind` source (value of `node.bind`) for each such source in the graph

For each `func_node`, `func_node.bind` gives us the
`{param: varnode_src_name}` specification that tells us where (key of scope)
to source the arguments of the `func_node.func` for each `param` of that
function.

What `params_for_src` is, is the corresponding inverse map.
The `{varnode_src_name: list_of_params}` gathered by scanning each
`func_node` of the DAG.

#### partial(\*positional_dflts, \_remove_bound_arguments=False, \_consider_defaulted_arguments_as_bound=False, \*\*keyword_dflts)

Get a curried version of the DAG.

Like `functools.partial`, but returns a DAG (not just a callable) and allows
you to remove bound arguments as well as roll in orphaned_nodes.

* **Parameters:**
  * **positional_dflts** – Bind arguments positionally
  * **keyword_dflts** – Bind arguments through their names
  * **\_remove_bound_arguments** – False – set to True if you don’t want bound
    arguments to show up in the signature.
  * **\_consider_defaulted_arguments_as_bound** – False – set to True if
    you want to also consider arguments that already had defaults as bound
    (and be removed).
* **Returns:**

```pycon
>>> def f(a, b):
...     return a + b
>>> def g(c, d=4):
...     return c * d
>>> def h(f, g):
...     return g - f
>>> dag = DAG([f, g, h])
>>> from inspect import signature
>>> str(signature(dag))
'(a, b, c, d=4)'
>>> dag(1, 2, 3, 4)  # == (3 * 4) - (1 + 2) == 12 - 3 == 9
9
>>> dag(c=3, a=1, b=2, d=4)  # same as above
9
```

```pycon
>>> new_dag = dag.partial(c=3)
>>> isinstance(new_dag, DAG)  # it's a dag (not just a partialized callable!)
True
>>> str(signature(new_dag))
'(a, b, c=3, d=4)'
>>> new_dag(1, 2)  # same as dag(c=3, a=1, b=2, d=4), so:
9
```

#### process_item(item)

Resolve a `slice` of node specifications into `(input_nodes, output_nodes)` lists, as used by `__getitem__`.

Each side of the slice may be `None` (all var nodes), a space-separated
string of names, a callable, or an iterable of names and callables; names are
resolved with `get_node_matching`. An `item` that is not a `slice` fails
an assertion, and a side of none of these forms raises `ValidationError`.

#### *property* sig

The DAG’s `__signature__` (an `i2.Sig`); assigning to it replaces
`__signature__`.

#### src_name_params(src_names=None)

Generate Parameter instances that are needed to compute `src_names`

#### synopsis_string(bind_info='var_nodes')

Join the `synopsis_string` of every func node, one per line in topological
order; `bind_info` controls how inputs are shown (see
`FuncNode.synopsis_string`).

#### to_code()

Render the DAG as the source of a function named after the DAG, one `out =
node_name(args)` line per func node (see `dag_to_code`).

### meshed.dag.arg_names(func, func_name, exclude_names=())

List the parameter names of `func`, replacing any found in `exclude_names` with
a free `<func_name>__<name>` variant.

### meshed.dag.attribute_vals(objs, attrs, egress=None)

Extract attributes from an iterable of objects

```pycon
>>> list(attribute_vals([print, map], attrs=['__name__', '__module__']))
[('print', 'builtins'), ('map', 'builtins')]
```

### meshed.dag.call_func(func, kwargs)

Re-key `kwargs` by each key’s `__name__` and pass the resulting dict to
`Sig(func).source_kwargs`.

### meshed.dag.change_value_on_cond(d, cond, func)

Replace, in place, each value `v` of `d` where `cond(k, v)` holds with
`func(v)`, and return `d`.

### meshed.dag.dag_to_code(dag)

Convert a DAG to code.

```pycon
>>> from meshed import code_to_dag
>>> @code_to_dag
... def dag():
...     a = func1(x, y)
...     b = func2(a, z)
...     c = func3(a, w=b)
>>>
```

Original DAG:

```pycon
>>> print(dag.synopsis_string())
x,y -> func1 -> a
a,z -> func2 -> b
a,b -> func3 -> c
```

Generated code using dag_to_code function:

```pycon
>>> code = dag_to_code(dag)
>>> print(code)
def dag():
    a = func1(x, y)
    b = func2(a, z)
    c = func3(a, w=b)
```

Test round-trip conversion:

```pycon
>>> dag2 = code_to_dag(code)
>>> print(dag2.synopsis_string())
x,y -> func1 -> a
a,z -> func2 -> b
a,b -> func3 -> c

>>> # Verify they're equivalent:
>>> dag.synopsis_string() == dag2.synopsis_string()
True
```

### meshed.dag.dflt_debugger_feedback(func_node, scope, output, step)

Print the step number, func node and scope, then return `output` unchanged
(default feedback of `DAG.debugger`).

### meshed.dag.find_first_free_name(prefix, exclude_names=(), start_at=2)

Return `prefix` if not in `exclude_names`, else the first free `prefix__<i>`
with `i` counting up from `start_at`.

### meshed.dag.funcnodes_from_pairs(pairs)

Make one mock func node per `(arg, out)` pair (see `mk_mock_funcnode`).

### meshed.dag.hook_up(func, variables, output_name=None)

Source inputs and write outputs to given variables mapping.

Returns inputless and outputless function that will, when called,
get relevant inputs from the provided variables mapping and write it’s
output there as well.

* **Parameters:**
  * **variables** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – The MutableMapping (like… a dict) where the function
    should both read it’s input and write it’s output.
  * **output_name** – The key of the variables mapping that should be used
    to write the output of the function
* **Returns:**
  A function

```pycon
>>> def formula1(w, /, x: float, y=1, *, z: int = 1):
...     return ((w + x) * y) ** z
```

```pycon
>>> d = {}
>>> f = hook_up(formula1, d)
>>> # NOTE: update d, not d = dict(...), which would make a DIFFERENT d
>>> d.update(w=2, x=3, y=4)  # not d = dict(w=2, x=3, y=4), which would
>>> f()
```

Note that there’s no output. The output is in d

```pycon
>>> d
{'w': 2, 'x': 3, 'y': 4, 'formula1': 20}
```

Again…

```pycon
>>> d.clear()
>>> d.update(w=1, x=2, y=3)
>>> f()
>>> d['formula1']
9
```

### meshed.dag.mk_func_name(func, exclude_names=())

Derive a name for `func` (its `__name__`, a generated lambda name, or the wrapped function’s name for a `partial`) that is not in `exclude_names`.

A `func` with no `__name__` that is not a `partial` raises `NameValidationError`.

### meshed.dag.mk_list_names_unique(nodes, exclude_names=())

List the `.name` of each node, suffixing repeats (and names in `exclude_names`)
with `__<i>` so all are distinct.

### meshed.dag.mk_mock_funcnode(arg, out)

Make a `FuncNode` whose no-op function takes the single parameter `arg` and
writes to `out`, named `_mock_<arg>_<out>`.

### meshed.dag.mk_nodes_names_unique(nodes)

Set each node’s `.name` in place to the unique names of `mk_list_names_unique`
and return `nodes`.

### meshed.dag.modified_func_node(func_node, \*\*modifications)

Make a new `FuncNode` from `func_node` with some of `func`, `name`, `bind`
and `out` replaced by `modifications`.

* **Return type:**
  [`FuncNode`](meshed.base.html.md#meshed.base.FuncNode)

### meshed.dag.named_partial(func, \*args, \_\_name_\_=None, \*\*keywords)

functools.partial, but with a \_\_name_\_

```pycon
>>> f = named_partial(print, sep='\n')
>>> f.__name__
'print'
```

```pycon
>>> f = named_partial(print, sep='\n', __name__='now_partial_has_a_name')
>>> f.__name__
'now_partial_has_a_name'
```

### meshed.dag.names_and_outs(objs, \*, attrs=('name', 'out'), egress=<class 'itertools.chain'>)

Extract attributes from an iterable of objects

```pycon
>>> list(attribute_vals([print, map], attrs=['__name__', '__module__']))
[('print', 'builtins'), ('map', 'builtins')]
```

### meshed.dag.order_subset_from_list(items, sublist)

Sort `sublist` by the position its elements have in `items`.

### meshed.dag.parametrized_dag_factory(dag, param_var_nodes)

Constructs a factory for sub-DAGs derived from the input DAG, with values of
specific ‘parameter’ variable nodes precomputed and fixed. These precomputed nodes,
and their ancestor nodes (unless required elsewhere), are omitted from the sub-DAG.

The factory function produced by this operation requires arguments corresponding to
the ancestor nodes of the parameter variable nodes. These arguments are used to
compute the values of the parameter nodes.

This function reflects the typical structure of a class in object-oriented
programming, where initialization arguments are used to set certain fixed values
(attributes), which are then leveraged in subsequent methods.

```pycon
>>> import i2
>>> from meshed import code_to_dag
>>> @code_to_dag
... def testdag():
...     a = criss(aa, aaa)
...     b = cross(aa, bb)
...     c = apple(a, b)
...     d = sauce(a, b)
...     e = applesauce(c, d)
>>>
>>> dag_factory = parametrized_dag_factory(testdag, 'a')
>>> print(f"{i2.Sig(dag_factory)}")
(aa, aaa)
>>> d = dag_factory(aa=1, aaa=2)
>>> print(f"{i2.Sig(d)}")
(b)
>>> d(b='bananna')
'applesauce(c=apple(a=criss(aa=1, aaa=2), b=bananna), d=sauce(a=criss(aa=1, aaa=2), b=bananna))'
```

### meshed.dag.partialized_funcnodes(func_nodes, \*\*keyword_defaults)

Yield the func nodes, replacing the function of any node whose parameters include a
`keyword_defaults` name with a partial where those parameters are defaulted and
moved last; other nodes are yielded as is.

### meshed.dag.print_dag_string(dag, bind_info='hybrid')

Print `dag.synopsis_string(bind_info)`; the default shows an input as
`param=var` only where the two names differ.

### meshed.dag.reorder_on_constraints(funcnodes, outs)

Topologically sort `funcnodes` after appending (in place) mock nodes that chain
each `outs` element to the next, print the order, and return `(func_nodes,
var_nodes)` without the mock nodes.
