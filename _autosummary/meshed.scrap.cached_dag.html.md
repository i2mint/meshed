# meshed.scrap.cached_dag

### Functions

| `add`(a[, b])                                                      |                                                                                                                                                                                       |
|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cached_dag_test`](#meshed.scrap.cached_dag.cached_dag_test)() | Covering issue [https://github.com/i2mint/meshed/issues/34](https://github.com/i2mint/meshed/issues/34) about "CachedDag.cache should be populated with inputs that it was called on" |
| `exp`(mult[, n])                                                   |                                                                                                                                                                                       |
| `func_node_names_and_outs`(dag)                                    |                                                                                                                                                                                       |
| `get_first_item_and_assert_unicity`(seq)                           |                                                                                                                                                                                       |
| `mult`(x[, y])                                                     |                                                                                                                                                                                       |
| `subtract`(a[, b])                                                 |                                                                                                                                                                                       |

### Classes

| [`CachedDag`](#meshed.scrap.cached_dag.CachedDag)(dag[, cache, name])   | Wraps a DAG, using it to compute any of it's var nodes from it's dependents, with the capability of caching intermediate var nodes for later reuse.   |
|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`NoOverwritesDict`](#meshed.scrap.cached_dag.NoOverwritesDict)                | A dict where you're not allowed to write to a key that already has a value in it.                                                                     |
| `NoSuchKey`()                                                                    |                                                                                                                                                       |

### Exceptions

| [`NotAllowed`](#meshed.scrap.cached_dag.NotAllowed)                | To use to indicate that something is not allowed              |
|----------------------------------------------------------------------------|---------------------------------------------------------------|
| [`OverWritesNotAllowedError`](#meshed.scrap.cached_dag.OverWritesNotAllowedError) | Error to raise when a writes to existing keys are not allowed |

### *class* meshed.scrap.cached_dag.CachedDag(dag, cache=True, name=None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

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

```w
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

We can get `ww` because it has a default:

(TODO: This (and further tests) stopped working since code_to_dag was enhanced
with the ability to use the wrapped function’s signature to determine the
signature of the output dag. Need to fix this.)

```pycon
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
>>> g('x', w=3)  # == 3 * 2 ==
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
{'x': 6}
```

and this will work too:

```pycon
>>> g('y')
7
>>> g.cache
{'x': 6, 'y': 7}
```

But this is something we need to handle better!

```pycon
>>> g('x', w=10)
6
```

This is happending because there’s already a x in the cache, and it takes precedence.
This would be okay if consider CachedDag as a low level object that is never
actually used by a user.
But we need to protect the user from such effects!

First, we probably should cache inputs too.

The we can:

- Make  computation take precedence over cache, overwriting the existing cache
  : with the new resulting values
- Allow the user to declare the entire cache, or just some variables in it,
  as write-once, to avoid creating bugs with the above proposal.
- Cache multiple paths (lru_cache style) for different input combinations

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

### *class* meshed.scrap.cached_dag.NoOverwritesDict

Bases: [`dict`](https://docs.python.org/3/library/stdtypes.html#dict)

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

### *exception* meshed.scrap.cached_dag.NotAllowed

Bases: [`Exception`](https://docs.python.org/3/library/exceptions.html#Exception)

To use to indicate that something is not allowed

### *exception* meshed.scrap.cached_dag.OverWritesNotAllowedError

Bases: [`NotAllowed`](#meshed.scrap.cached_dag.NotAllowed)

Error to raise when a writes to existing keys are not allowed

### meshed.scrap.cached_dag.cached_dag_test()

Covering issue [https://github.com/i2mint/meshed/issues/34](https://github.com/i2mint/meshed/issues/34)
about “CachedDag.cache should be populated with inputs that it was called on”
