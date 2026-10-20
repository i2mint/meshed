# meshed.cached_dag

Compute any var node of a DAG from what it depends on, caching what was computed.

[`CachedDag`](#meshed.cached_dag.CachedDag) wraps a [`DAG`](meshed.dag.html.md#meshed.dag.DAG): `cached(k, **inputs)` computes the
var node `k` from the given inputs, the cache, and the defaults, and keeps every input
and intermediate value for later calls. [`CachedDag.inject_methods()`](#meshed.cached_dag.CachedDag.inject_methods) gives an object
one method per computable var node, whose signature lists just the root inputs that
node still needs (i2mint/meshed#36).

```pycon
>>> from meshed import DAG
>>> def f(a, x=1):
...     return a + x
>>> def g(a, y=2):
...     return a * y
>>> cached = CachedDag(DAG([f, g]))
>>> cached('g', a=1), cached('f')
(2, 2)
```

### Classes

| [`CachedDag`](#meshed.cached_dag.CachedDag)(dag[, cache, name])   | Wraps a DAG, using it to compute any of it's var nodes from it's dependents, with the capability of caching intermediate var nodes for later reuse.   |
|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`NoOverwritesDict`](#meshed.cached_dag.NoOverwritesDict)                | A dict where you're not allowed to write to a key that already has a value in it.                                                                     |

### Exceptions

| [`NotAllowed`](#meshed.cached_dag.NotAllowed)                | To use to indicate that something is not allowed              |
|----------------------------------------------------------------------------|---------------------------------------------------------------|
| [`OverWritesNotAllowedError`](#meshed.cached_dag.OverWritesNotAllowedError) | Error to raise when a writes to existing keys are not allowed |

### *class* meshed.cached_dag.CachedDag(dag, cache=True, name=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wraps a DAG, using it to compute any of it’s var nodes from it’s dependents,
with the capability of caching intermediate var nodes for later reuse.

```pycon
>>> def add(a, b=1):
...     return a + b
>>> def mult(x, y=2):
...     return x * y
>>> def subtract(a, b=4):
...     return a - b
>>> from meshed import code_to_dag
>>>
>>> @code_to_dag(func_src=locals())
... def dag(w, ww, www):
...     x = mult(w, ww)
...     y = add(x, www)
...     z = subtract(x, y)
>>> print(dag.dot_digraph_ascii())
```

```text
               w

            │
            │
            ▼
          ┌──────────┐
ww=   ──▶ │   mult   │
          └──────────┘
            │
            │
            ▼

               x       ─┐
                        │
            │           │
            │           │
            ▼           │
          ┌──────────┐  │
www=  ──▶ │   add    │  │
          └──────────┘  │
            │           │
            │           │
            ▼           │
                        │
               y=       │
                        │
            │           │
            │           │
            ▼           │
          ┌──────────┐  │
          │ subtract │ ◀┘
          └──────────┘
            │
            │
            ▼

               z
```

```pycon
>>> from inspect import signature
>>> g = CachedDag(dag)
>>> signature(g)
<Signature (k, /, **input_kwargs)>
```

We can get `ww` because it has a default: the dag’s own signature
(`w, ww, www`) has none, but `ww` feeds `mult`’s `y=2`, and a root takes
the default of the parameters it binds (when they agree):

```pycon
>>> g.defaults
{'ww': 2, 'www': 1}
>>> g('ww')
2
```

But we can’t get `y` because we don’t have what it depends on:

```pycon
>>> g('y')
Traceback (most recent call last):
    ...
TypeError: The input_kwargs of a dag call is missing 1 required argument: 'w'
```

It needs a `w?`! No, it needs an `x`! But to get an `x` you need a `w`,
and…

```pycon
>>> g('x')
Traceback (most recent call last):
    ...
TypeError: The input_kwargs of a dag call is missing 1 required argument: 'w'
```

So let’s give it a w!

```pycon
>>> g('x', w=3)  # == 3 * 2
6
```

And now this works:

```pycon
>>> g('x')
6
```

because

```pycon
>>> g.cache
{'x': 6, 'w': 3}
```

and this will work too:

```pycon
>>> g('y')
7
>>> g.cache
{'x': 6, 'w': 3, 'y': 7}
```

An input that contradicts the cache is refused rather than silently ignored:

```pycon
>>> g('x', w=10)
Traceback (most recent call last):
    ...
ValueError: input_kwargs can't contain keys that are already in cache with a different value! These names were in both: {'w'}
```

Methods per var node, each asking only for the roots it still needs:

```pycon
>>> fresh = CachedDag(dag).inject_methods()
>>> signature(fresh.y)
<Signature (*, w, ww=2, www=1)>
>>> fresh.x(w=3)
6
>>> signature(fresh.y)  # w is cached now
<Signature (*, ww=2, www=1)>
>>> fresh.z()
-1
```

#### inject_methods(obj=None)

Give `obj` (default: a new namespace) one method per non-root var node.

`obj.<node>(**roots)` is `self(<node>, **roots)`. Its signature is computed
when asked, so it always lists the roots `<node>` still needs given the cache
at that moment, with the defaults the dag (or the nodes it feeds) gives them.

#### roots_for(node)

The set of roots that lead to `node`.

```pycon
>>> from meshed.makers import code_to_dag
>>> @code_to_dag
... def dag():
...     x = mult(w, ww)
...     y = add(x, www)
...     z = subtract(x, y)
>>> print(dag.synopsis_string())
w,ww -> mult -> x
x,www -> add -> y
x,y -> subtract -> z
>>> g = CachedDag(dag)
>>> sorted(g.roots_for('x'))
['w', 'ww']
>>> sorted(g.roots_for('y'))
['w', 'ww', 'www']
```

### *class* meshed.cached_dag.NoOverwritesDict

Bases: [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

A dict where you’re not allowed to write to a key that already has a value in it.

```pycon
>>> d = NoOverwritesDict(a=1, b=2)
>>> d
{'a': 1, 'b': 2}
```

Writing is allowed, in new keys

```pycon
>>> d['c'] = 3
>>> d
{'a': 1, 'b': 2, 'c': 3}
```

It’s also okay to write into an existing key if the value it holds is identical.
In fact, the write doesn’t even happen.

```pycon
>>> d['b'] = 2
```

But if we try to write a different value…

```pycon
>>> d['b'] = 22
Traceback (most recent call last):
    ...
cached_dag.OverWritesNotAllowedError: The b key already exists and you're not allowed to change its value
```

### *exception* meshed.cached_dag.NotAllowed

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

To use to indicate that something is not allowed

### *exception* meshed.cached_dag.OverWritesNotAllowedError

Bases: [`NotAllowed`](#meshed.cached_dag.NotAllowed)

Error to raise when a writes to existing keys are not allowed
