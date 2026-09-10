# meshed.util

util functions

### Functions

| `arg_names`(func, func_name[, exclude_names])                                                      |                                                                                                                                                                       |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`args_funcnames`](#meshed.util.args_funcnames)(funcs[, name_of_func])             | Generates (arg_name, func_id) pairs from the iterable of functions                                                                                                    |
| [`conditional_trans`](#meshed.util.conditional_trans)(obj, condition, trans)          | Conditionally transform an object unless it is marked as a literal.                                                                                                   |
| `curry`(func)                                                                                      |                                                                                                                                                                       |
| [`dot_to_ascii`](#meshed.util.dot_to_ascii)(dot[, fancy])                        | Convert a dot string to an ascii rendering of the diagram.                                                                                                            |
| `extra_wraps`(func[, name, doc_prefix])                                                            |                                                                                                                                                                       |
| [`extract_dict`](#meshed.util.extract_dict)(d, keys)                             | Extract items from dict `d`, returning them as a dict.                                                                                                                |
| [`extract_items`](#meshed.util.extract_items)(d, keys)                            | generator of (k, v) pairs extracted from d for keys                                                                                                                   |
| [`extract_values`](#meshed.util.extract_values)(d, keys)                           | Extract values from dict `d`, returning them:                                                                                                                         |
| [`filepath_to_module`](#meshed.util.filepath_to_module)(file_path)                     | A context manager to import a Python file as a module.                                                                                                                |
| `find_first_free_name`(prefix[, ...])                                                              |                                                                                                                                                                       |
| [`func_name`](#meshed.util.func_name)(func)                                   | The func._\_name_\_ of a callable func, or makes and returns one if that fails.                                                                                       |
| [`funcs_conjunction`](#meshed.util.funcs_conjunction)(\*funcs)                        | Makes a conjunction of functions.                                                                                                                                     |
| [`funcs_disjunction`](#meshed.util.funcs_disjunction)(\*funcs)                        | Makes a disjunction of functions.                                                                                                                                     |
| `funcs_to_digraph`(funcs[, graph])                                                                 |                                                                                                                                                                       |
| [`if_then_else`](#meshed.util.if_then_else)(if_func, then_func, else_func, ...)  | Tool to "functionalize" the if-then-else logic.                                                                                                                       |
| [`incremental_str_maker`](#meshed.util.incremental_str_maker)([str_format])               | Make a function that will produce a (incrementally) new string at every call.                                                                                         |
| [`instance_checker`](#meshed.util.instance_checker)(class_or_tuple)                  | Makes a boolean function that checks the instance of an object                                                                                                        |
| `inverse_dict_asserting_losslessness`(d)                                                           |                                                                                                                                                                       |
| [`iterize`](#meshed.util.iterize)(func[, name])                             | From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output] Some call this "vectorization", but it's not really a vector, but an iterable, thus the name. |
| `lambda_name`()                                                                                    |                                                                                                                                                                       |
| [`mk_func_name`](#meshed.util.mk_func_name)(func[, exclude_names])               | Makes a function name that doesn't clash with the exclude_names iterable.                                                                                             |
| [`mk_place_holder_func`](#meshed.util.mk_place_holder_func)(arg_names_or_sig[, ...])     | Make (working and picklable) function with a specific signature.                                                                                                      |
| [`my_isinstance`](#meshed.util.my_isinstance)(obj, class_or_tuple)                | Same as builtin instance, but without position only constraint.                                                                                                       |
| `mywraps`(func[, name, doc_prefix])                                                                |                                                                                                                                                                       |
| [`named_partial`](#meshed.util.named_partial)(func, \*args[, \_\_name_\_])        | functools.partial, but with a \_\_name_\_                                                                                                                             |
| [`numbered_suffix_renamer`](#meshed.util.numbered_suffix_renamer)(name[, sep])              |                                                                                                                                                                       |
| [`objects_defined_in_module`](#meshed.util.objects_defined_in_module)(module, \*[, ...])      | Get a dictionary of objects defined in a Python module, optionally filtered by their names and values.                                                                |
| [`ordered_set_operations`](#meshed.util.ordered_set_operations)(a, b)                      | Returns a triple (a-b, a&b, b-a) for two iterables a and b.                                                                                                           |
| `pairs`(xs)                                                                                        |                                                                                                                                                                       |
| [`parameter_merger`](#meshed.util.parameter_merger)(\*params[, same_name, ...])      | Validates that all the params are exactly the same, returning the first if so.                                                                                        |
| `print_ascii_graph`(funcs)                                                                         |                                                                                                                                                                       |
| [`provides`](#meshed.util.provides)(\*var_names)                             | Decorator to assign `var_names` to a `_provides` attribute of function.                                                                                               |
| [`replace_item_in_iterable`](#meshed.util.replace_item_in_iterable)(iterable, ...[, egress]) | Returns a list where all items satisfying `condition(item)` were replaced with `replacement(item)`.                                                                   |
| `uncurry`(func)                                                                                    |                                                                                                                                                                       |
| `unnameable_func_name`()                                                                           |                                                                                                                                                                       |

### Classes

| [`ConditionalIterize`](#meshed.util.ConditionalIterize)(func[, iterize_type, ...])   | A decorator that "iterizes" a function call if input satisfies a condition.   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `ModuleNotFoundIgnore`()                                                                         |                                                                               |

### Exceptions

| [`InvalidFunctionParameters`](#meshed.util.InvalidFunctionParameters)   | To be used when a function's parameters are not compliant with some rule about them.   |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [`NameValidationError`](#meshed.util.NameValidationError)         | Use to indicate that there's a problem with a name or generating a valid name          |
| [`NotFound`](#meshed.util.NotFound)                    | To be raised when something is expected to exist, but doesn't                          |
| [`NotUniqueError`](#meshed.util.NotUniqueError)              | Error to be raised when unicity is expected, but violated                              |
| [`ValidationError`](#meshed.util.ValidationError)             | Error that is raised when an object's validation failed                                |

### *class* meshed.util.ConditionalIterize(func, iterize_type=<class 'collections.abc.Iterator'>, iterize_condition=None)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

A decorator that “iterizes” a function call if input satisfies a condition.
That is, apply `map(func, input)` (iterize) or `func(input)` according to some
conidition on `input`.

```pycon
>>> def foo(x, y=2):
...     return x * y
```

The function does this:

```pycon
>>> foo(3)
6
>>> foo('string')
'stringstring'
```

The iterized version of the function does this:

```pycon
>>> iterized_foo = iterize(foo)
>>> list(iterized_foo([1, 2, 3]))
[2, 4, 6]
```

```pycon
>>> from typing import Iterable
>>> new_foo = ConditionalIterize(foo, Iterable)
>>> new_foo(3)
6
>>> list(new_foo([1, 2, 3]))
[2, 4, 6]
```

See what happens if we do this:

```pycon
>>> list(new_foo('string'))
['ss', 'tt', 'rr', 'ii', 'nn', 'gg']
```

Maybe you expected `'stringstring'` because you are thinking of `string` as a valid,
single input. But the condition of iterization is to be an Iterable, which a
string is, thus the (perhaps) unexpected result.

In fact, this problem is a general one:
If your base function doesn’t process iterables, the `isinstance(x, Iterable)`
is good enough – but if it is supposed to process an iterable in the first place,
how can you distinguish whether to use the iterized version or not?
The solution depends on the situation and the iterface you want. You choose.

Since the situation where you’ll want to iterize functions in the first place is when
you’re building streaming pipelines, a good fallback choice is to iterize if and
only if the input is an iterator. This is condition will trigger the iterization
when the input has a `__next__` – so things like generators, but not lists,
tuples, sets, etc.

See in the following that `ConditionalIterize` also has a `wrap` class method
that can be used to wrap a function at definition time.

```pycon
>>> @ConditionalIterize.wrap(Iterator)  # Iterator is the default, so no need here
... def foo(x, y=2):
...     return x * y
>>> foo(3)
6
>>> foo('string')
'stringstring'
```

If you want to process a “stream” of numbers 1, 2, 3, don’t do it this way:

```pycon
>>> foo([1, 2, 3])
[1, 2, 3, 1, 2, 3]
```

Instead, you should explicitly wrap that iterable in an iterator, to trigger the
iterization:

```pycon
>>> list(foo(iter([1, 2, 3])))
[2, 4, 6]
```

So far, the only way we controlled the iterize condition is through a type.
Really, the condition that is used behind the scenes is
`isinstance(obj, self.iterize_type)`.
If you need more complex conditions though, you can specify it through the
`iterize_condition` argument. The `iterize_type` is also used to
annotate the resulting wrapped function if it’s first argument is annotated.
As a consequence, `iterize_type` needs to be a “generic” type.

```pycon
>>> @ConditionalIterize.wrap(Iterable, lambda x: isinstance(x, (list, tuple)))
... def foo(x: int, y=2):
...     return x * y
>>> foo(3)
6
>>> list(foo([1, 2, 3]))
[2, 4, 6]
>>> from inspect import signature
```

We annotated `x` as `int`, so see now the annotation of the wrapped function:

```pycon
>>> str(signature(foo))
'(x: Union[int, Iterable[int]], y=2)'
```

### *exception* meshed.util.InvalidFunctionParameters

Bases: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

To be used when a function’s parameters are not compliant with some rule about
them.

### *exception* meshed.util.NameValidationError

Bases: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

Use to indicate that there’s a problem with a name or generating a valid name

### *exception* meshed.util.NotFound

Bases: [`ValidationError`](#meshed.util.ValidationError)

To be raised when something is expected to exist, but doesn’t

### *exception* meshed.util.NotUniqueError

Bases: [`ValidationError`](#meshed.util.ValidationError)

Error to be raised when unicity is expected, but violated

### *exception* meshed.util.ValidationError

Bases: [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

Error that is raised when an object’s validation failed

### meshed.util.args_funcnames(funcs, name_of_func=<function func_name>)

Generates (arg_name, func_id) pairs from the iterable of functions

### meshed.util.conditional_trans(obj, condition, trans)

Conditionally transform an object unless it is marked as a literal.

```pycon
>>> from functools import partial
>>> trans = partial(
...     conditional_trans, condition=str.isnumeric, trans=float
... )
>>> trans('not a number')
'not a number'
>>> trans('10')
10.0
```

To use this function but tell it to not transform some a specific input no matter
what, wrap the input with `Literal`

```pycon
>>> # from meshed import Literal
>>> conditional_trans(LiteralVal('10'), str.isnumeric, float)
'10'
```

### meshed.util.conservative_parameter_merge(\*params, same_name=True, same_kind=True, same_default=True, same_annotation=True)

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

### meshed.util.dot_to_ascii(dot, fancy=True)

Convert a dot string to an ascii rendering of the diagram.

Needs a connection to the internet to work.

```pycon
>>> graph_dot = '''
...     graph {
...         rankdir=LR
...         0 -- {1 2}
...         1 -- {2}
...         2 -> {0 1 3}
...         3
...     }
... '''
>>>
>>> graph_ascii = dot_to_ascii(graph_dot)
>>>
>>> print(graph_ascii)

                 ┌─────────┐
                 ▼         │
     ┌───┐     ┌───┐     ┌───┐     ┌───┐
  ┌▶ │ 0 │ ─── │ 1 │ ─── │   │ ──▶ │ 3 │
  │  └───┘     └───┘     │   │     └───┘
  │    │                 │   │
  │    └──────────────── │ 2 │
  │                      │   │
  │                      │   │
  └───────────────────── │   │
                         └───┘
```

### meshed.util.extract_dict(d, keys)

Extract items from dict `d`, returning them as a dict.

```pycon
>>> extract_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
{'a': 1, 'c': 3}
```

Order matters!

```pycon
>>> extract_dict({'a': 1, 'b': 2, 'c': 3}, ['c', 'a'])
{'c': 3, 'a': 1}
```

### meshed.util.extract_items(d, keys)

generator of (k, v) pairs extracted from d for keys

```pycon
>>> list(extract_items({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']))
[('a', 1), ('c', 3)]
```

### meshed.util.extract_values(d, keys)

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

### meshed.util.filepath_to_module(file_path)

A context manager to import a Python file as a module.

* **Parameters:**
  **file_path** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The file path of the Python file to import.
* **Yield:**
  The module object.

### meshed.util.func_name(func)

The func._\_name_\_ of a callable func, or makes and returns one if that fails.
To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing

* **Return type:**
  [`str`](https://docs.python.org/3/library/stdtypes.html#str)

### meshed.util.funcs_conjunction(\*funcs)

Makes a conjunction of functions. That is, `func1(x) and func2(x) and ...`

```pycon
>>> f = funcs_conjunction(lambda x: isinstance(x, str), lambda x: len(x) >= 5)
>>> f('app')  # because length is less than 5...
False
>>> f('apple')  # length at least 5 so...
True
```

Note that in:

```pycon
>>> f(42)
False
```

it is `False` because it is not a string.
This shows that the second function is not applied to the input at all, since it
doesn’t need to, and if it were, we’d get an error (length of a number?!).

### meshed.util.funcs_disjunction(\*funcs)

Makes a disjunction of functions. That is, `func1(x) or func2(x) or ...`

```pycon
>>> f = funcs_disjunction(lambda x: x > 10, lambda x: x < -5)
>>> f(7)
False
>>> f(-7)
True
```

### meshed.util.if_then_else(if_func, then_func, else_func, \*args, \*\*kwargs)

Tool to “functionalize” the if-then-else logic.

```pycon
>>> from functools import partial
>>> f = partial(if_then_else, str.isnumeric, int, str)
>>> f('a string')
'a string'
>>> f('42')
42
```

### meshed.util.incremental_str_maker(str_format='{:03.f}')

Make a function that will produce a (incrementally) new string at every call.

### meshed.util.instance_checker(class_or_tuple)

Makes a boolean function that checks the instance of an object

```pycon
>>> isinstance_of_str = instance_checker(str)
>>> isinstance_of_str('asdf')
True
>>> isinstance_of_str(3)
False
```

### meshed.util.iterize(func, name=None)

From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output]
Some call this “vectorization”, but it’s not really a vector, but an
iterable, thus the name.

`iterize` is a partial of `map`.

```pycon
>>> f = lambda x: x * 10
>>> f(2)
20
>>> iterized_f = iterize(f)
>>> list(iterized_f(iter([1,2,3])))
[10, 20, 30]
```

Consider the following pipeline:

```pycon
>>> from i2 import Pipe
>>> pipe = Pipe(lambda x: x * 2, lambda x: f"hello {x}")
>>> pipe(1)
'hello 2'
```

But what if you wanted to use the pipeline on a “stream” of data. The
following wouldn’t work:

```pycon
>>> try:
...     pipe(iter([1,2,3]))
... except TypeError as e:
...     print(f"{type(e).__name__}: {e}")
...
...
TypeError: unsupported operand type(s) for *: 'list_iterator' and 'int'
```

Remember that error: You’ll surely encounter it at some point.

The solution to it is (often): `iterize`,
which transforms a function that is meant to be applied to a single object,
into a function that is meant to be applied to an array, or any iterable
of such objects.
(You might be familiar (if you use `numpy` for example) with the related
concept of “vectorization”,
or [array programming](https://en.wikipedia.org/wiki/Array_programming).)

```pycon
>>> from i2 import Pipe
>>> from meshed.util import iterize
>>> from typing import Iterable
>>>
>>> pipe = Pipe(
...     iterize(lambda x: x * 2),
...     iterize(lambda x: f"hello {x}")
... )
>>> iterable = pipe([1, 2, 3])
>>> # see that the result is an iterable
>>> assert isinstance(iterable, Iterable)
>>> list(iterable)  # consume the iterable and gather it's items
['hello 2', 'hello 4', 'hello 6']
```

### meshed.util.mk_func_name(func, exclude_names=())

Makes a function name that doesn’t clash with the exclude_names iterable.
Tries it’s best to not be lazy, but instead extract a name from the function
itself.

### meshed.util.mk_place_holder_func(arg_names_or_sig, name=None, defaults=(), annotations=())

Make (working and picklable) function with a specific signature.

This is useful for testing as well as injecting compliant functions in DAG templates.

* **Parameters:**
  * **arg_names_or_sig** – Anything that i2.Sig can accept as it’s first input.
    (Such as a string of argument(s), function, signature, etc.)
  * **name** – The `__name__` to give the function.
  * **defaults** – If you want to add/change defaults
  * **annotations** – If you want to add/change annotations
* **Returns:**
  A (working and picklable) function with a specific signature

```pycon
>>> f = mk_place_holder_func('a b', 'my_func')
>>> f(1,2)
'my_func(a=1, b=2)'
```

The first argument can be any expression of a signature that `i2.Sig` can
understand. For instance, it could be a function itself.
See how the function takes on `mk_place_holder_func`’s signature and name in the
following example:

```pycon
>>> g = mk_place_holder_func(mk_place_holder_func)
>>> from inspect import signature
>>> str(signature(g))  # should give the same signature as mk_place_holder_func
'(arg_names_or_sig, name=None, defaults=(), annotations=())'
>>> g(1,2,defaults=3, annotations=4)
'mk_place_holder_func(arg_names_or_sig=1, name=2, defaults=3, annotations=4)'
```

### meshed.util.my_isinstance(obj, class_or_tuple)

Same as builtin instance, but without position only constraint.
Therefore, we can partialize class_or_tuple:

Otherwise, couldn’t do:

```pycon
>>> isinstance_of_str = partial(my_isinstance, class_or_tuple=str)
>>> isinstance_of_str('asdf')
True
>>> isinstance_of_str(3)
False
```

### meshed.util.named_partial(func, \*args, \_\_name_\_=None, \*\*keywords)

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

### meshed.util.numbered_suffix_renamer(name, sep='_')

```pycon
>>> numbered_suffix_renamer('item')
'item_1'
>>> numbered_suffix_renamer('item_1')
'item_2'
```

### meshed.util.objects_defined_in_module(module, , name_filt=None, obj_filt=None)

Get a dictionary of objects defined in a Python module, optionally filtered by their names and values.

* **Parameters:**
  * **module** ([`str`](https://docs.python.org/3/library/stdtypes.html#str) | [`ModuleType`](https://docs.python.org/3/library/types.html#types.ModuleType)) – 

    The module to look up. Can either be
    - the module object itself,
    - a string specifying the module’s fully qualified name (e.g., ‘os.path’), or
    - a .py filepath to the module
  * **name_filt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/library/constants.html#None)) – An optional function used to filter the names of objects in the module.
    This function should take a single argument (the object name as a string)
    and return a boolean. Only objects whose names pass the filter (i.e.,
    for which the function returns True) are included.
    If None, no name filtering is applied.
  * **obj_filt** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/library/constants.html#None)) – An optional function used to filter the objects in the module. This function should take a
    single argument (the object itself) and return a boolean. Only objects that pass the filter
    (i.e., for which the function returns True) are included.
    If None, no object filtering is applied.
* **Returns:**
  A dictionary where keys are names of objects defined in the module (filtered by name_filt and obj_filt)
  and values are the corresponding objects.
* **Return type:**
  [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)

### Examples

```pycon
>>> import os
>>> all_os_objects = objects_defined_in_module(os)
>>> 'removedirs' in all_os_objects
True
>>> all_os_objects['removedirs'] == os.removedirs
True
```

See that you can specify the module via a string too, and filter to get only
callables that don’t start with an underscore:

```pycon
>>> this_modules_funcs = objects_defined_in_module(
...     'meshed.util',
...     name_filt=lambda name: not name.startswith('_'),
...     obj_filt=callable,
... )
>>> callable(this_modules_funcs['objects_defined_in_module'])
True
```

### meshed.util.ordered_set_operations(a, b)

Returns a triple (a-b, a&b, b-a) for two iterables a and b.
The operations are performed as if a and b were sets, but the order in a is conserved.

* **Return type:**
  [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple)[[`list`](https://docs.python.org/3/library/stdtypes.html#list), [`list`](https://docs.python.org/3/library/stdtypes.html#list), [`list`](https://docs.python.org/3/library/stdtypes.html#list)]

```pycon
>>> ordered_set_operations([1, 2, 3, 4], [3, 4, 5, 6])
([1, 2], [3, 4], [5, 6])
```

```pycon
>>> ordered_set_operations("abcde", "cdefg")
(['a', 'b'], ['c', 'd', 'e'], ['f', 'g'])
```

```pycon
>>> ordered_set_operations([1, 2, 2, 3], [2, 3, 3, 4])
([1], [2, 3], [4])
```

### meshed.util.parameter_merger(\*params, same_name=True, same_kind=True, same_default=True, same_annotation=True)

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

### meshed.util.provides(\*var_names)

Decorator to assign `var_names` to a `_provides` attribute of function.

This is meant to be used to indicate to a mesh what var nodes a function can source
values for.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)], [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]

```pycon
>>> @provides('a', 'b')
... def f(x):
...     return x + 1
>>> f._provides
('a', 'b')
```

If no `var_names` are given, then the function name is used as the var name:

```pycon
>>> @provides()
... def g(x):
...     return x + 1
>>> g._provides
('g',)
```

If `var_names` contains `'_'`, then the function name is used as the var name
for that position:

```pycon
>>> @provides('b', '_')
... def h(x):
...     return x + 1
>>> h._provides
('b', 'h')
```

### meshed.util.replace_item_in_iterable(iterable, condition, replacement, , egress=None)

Returns a list where all items satisfying `condition(item)` were replaced
with `replacement(item)`.

If `condition` is not a callable, it will be considered as a value to check
against using `==`.

If `replacement` is not a callable, it will be considered as the actual
value to replace by.

* **Parameters:**
  * **iterable** – Input iterable of items
  * **condition** – Condition to apply to item to see if it should be replaced
  * **replacement** – (Conditional) replacement value or function
  * **egress** – The function to apply to transformed iterable

```pycon
>>> replace_item_in_iterable([1,2,3,4,5], condition=2, replacement = 'two')
[1, 'two', 3, 4, 5]
>>> is_even = lambda x: x % 2 == 0
>>> replace_item_in_iterable([1,2,3,4,5], condition=is_even, replacement = 'even')
[1, 'even', 3, 'even', 5]
>>> replace_item_in_iterable([1,2,3,4,5], is_even, replacement=lambda x: x * 10)
[1, 20, 3, 40, 5]
```

Note that if the input iterable is not a `list`, `tuple`, or `set`,
your output will be an iterator that you’ll have to iterate through to gather
transformed items.

```pycon
>>> g = replace_item_in_iterable(iter([1,2,3,4,5]), condition=2, replacement = 'two')
>>> isinstance(g, Iterator)
True
```

Unless you specify an egress of your choice:

```pycon
>>> replace_item_in_iterable(
... iter([1,2,3,4,5]), is_even, lambda x: x * 10, egress=sorted
... )
[1, 3, 5, 20, 40]
```
