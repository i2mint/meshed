# meshed.base

Base functionality of meshed

### Functions

| [`basic_node_validator`](#meshed.base.basic_node_validator)(func_node)                  | Validates a func node.                                                                     |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`ch_func_node_attrs`](#meshed.base.ch_func_node_attrs)(fn, \*\*new_attrs_values)     | Returns a copy of the func node with some of its attributes changed                        |
| `ch_func_node_func`(fn, func, \*[, ...])                                                          |                                                                                            |
| `dot_lines_of_func_parameters`(parameters, ...)                                                   |                                                                                            |
| `duplicates`(elements)                                                                            |                                                                                            |
| [`ensure_func_nodes`](#meshed.base.ensure_func_nodes)(func_nodes)                    | Converts a list of objects to a list of FuncNodes.                                         |
| [`func_node_transformer`](#meshed.base.func_node_transformer)(fn[, kwargs_transformers]) | Get a modified `FuncNode` from an iterable of `kwargs_trans` modifiers.                    |
| [`func_nodes_to_code`](#meshed.base.func_nodes_to_code)(func_nodes[, func_name, ...]) | Convert an iterable of FuncNodes back to executable Python code.                           |
| [`get_init_params_of_instance`](#meshed.base.get_init_params_of_instance)(obj)                 | Get names of instance object `obj` that are also parameters of the `__init__` of its class |
| `handle_variadics`(func)                                                                          |                                                                                            |
| [`identifier_mapping`](#meshed.base.identifier_mapping)(x)                            | Get an `IdentifierMapping` dict from a more loosely defined `Bind`.                        |
| `insert_func_if_compatible`([func_comparator])                                                    |                                                                                            |
| [`is_func_node`](#meshed.base.is_func_node)(obj)                                |                                                                                            |
| [`is_not_func_node`](#meshed.base.is_not_func_node)(obj)                            |                                                                                            |
| `param_to_dot_definition`(p[, shape])                                                             |                                                                                            |
| `raise_signature_mismatch_error`(fn, func)                                                        |                                                                                            |
| [`rebind_to_func`](#meshed.base.rebind_to_func)(fnode, new_func)                  | Replaces `fnode.func` with `new_func`, changing the `.bind` accordingly.                   |
| [`underscore_func_node_names_maker`](#meshed.base.underscore_func_node_names_maker)(func[, ...])    | This name maker will resolve names in the following fashion:                               |
| [`validate_that_func_node_names_are_sane`](#meshed.base.validate_that_func_node_names_are_sane)(...)      | Assert that the names of func_nodes are sane.                                              |

### Classes

| [`FuncNode`](#meshed.base.FuncNode)(func[, name, bind, out, ...])   | A function wrapper that makes the function amenable to operating in a network.   |
|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`Mesh`](#meshed.base.Mesh)(func_nodes)                         |                                                                                  |

### *class* meshed.base.FuncNode(func, name=None, bind=<factory>, out=None, func_label=None, names_maker=<function underscore_func_node_names_maker>, node_validator=<function basic_node_validator>)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

A function wrapper that makes the function amenable to operating in a network.

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Function to wrap
  * **name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The name to associate to the function
  * **bind** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict)) – The {func_argname: external_name,…} mapping that defines where
    the node will source the data to call the function.
    This only has to be used if the external names are different from the names
    of the arguments of the function.
  * **out** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – The variable name the function should write it’s result to

Like we stated: `FuncNode` is meant to operate in computational networks.
But knowing what it does will help you make the networks you want, so we commend
your curiousity, and will oblige with an explanation.

Say you have a function to multiply numbers.

```pycon
>>> def multiply(x, y):
...     return x * y
```

And you use it in some code like this:

```pycon
>>> item_price = 3.5
>>> num_of_items = 2
>>> total_price = multiply(item_price, num_of_items)
```

What the execution of `total_price = multiply(item_price, num_of_items)` does is

- grab the values (in the locals scope – a dict), of `item_price` and `num_of_items`,
- call the multiply function on these, and then
- write the result to a variable (in locals) named `total_price`

`FuncNode` is a function wrapper that specification of such a
`output = function(...inputs...)` assignment statement
in such a way that it can carry it out on a `scope`.
A `scope` is a `dict` where the function can find it’s input values and write its
output values.

For example, the `FuncNode` form of the above statement would be:

```pycon
>>> func_node = FuncNode(
...     func=multiply,
...     bind={'x': 'item_price', 'y': 'num_of_items'})
>>> func_node
FuncNode(x=item_price,y=num_of_items -> multiply_ -> multiply)
```

Note the `bind` is a mapping **from** the variable names of the wrapped function
**to** the names of the scope.

That is, when it’s time to execute, it tells the `FuncNode` where to find the values
of its inputs.

If an input is not specified in this `bind` mapping, the scope
(external) name is supposed to be the same as the function’s (internal) name.

The purpose of a `FuncNode` is to source some inputs somewhere, compute something
with these, and write the result somewhere. That somewhere is what we call a
scope. A scope is a dictionary (or any mutuable mapping to be precise) and it works
like this:

```pycon
>>> scope = {'item_price': 3.5, 'num_of_items': 2}
>>> func_node.call_on_scope(scope)  # see that it returns 7.0
7.0
>>> scope  # but also wrote this in the scope
{'item_price': 3.5, 'num_of_items': 2, 'multiply': 7.0}
```

Consider `item_price,num_of_items -> multiply_ -> multiply`.
See that the name of the function is used for the name of its output,
and an underscore-suffixed name for its function name.
That’s the default behavior if you don’t specify either a name (of the function)
for the `FuncNode`, or a `out`.
The underscore is to distinguish from the name of the function itself.
The function gets the underscore because this favors particular naming style.

You can give it a custom name as well.

```pycon
>>> FuncNode(multiply, name='total_price', out='daily_expense')
FuncNode(x,y -> total_price -> daily_expense)
```

If you give an `out`, but not a `name` (for the function), the function’s
name will be taken:

```pycon
>>> FuncNode(multiply, out='daily_expense')
FuncNode(x,y -> multiply -> daily_expense)
```

If you give a `name`, but not a `out`, an underscore-prefixed version of
the `name` will be taken:

```pycon
>>> FuncNode(multiply, name='total_price')
FuncNode(x,y -> total_price -> _total_price)
```

#### NOTE
In the context of networks if you want to reuse a same function
(say, `multiply`) in multiple places
you’ll **need** to give it a custom name because the functions are identified by
this name in the network.

#### call_on_scope(scope, write_output_into_scope=True)

Call the function using the given scope both to source arguments and write
results.

#### NOTE
This method is only meant to be used as a backend to \_\_call_\_, not as
an actual interface method. Additional control/constraints on read and writes
can be implemented by providing a custom scope for that.

#### ch_attrs(\*\*new_attrs_values)

Returns a copy of the func node with some of its attributes changed

```pycon
>>> def plus(a, b):
...     return a + b
...
>>> def minus(a, b):
...     return a - b
...
>>> fn = FuncNode(func=plus, out='sum')
>>> fn.func == plus
True
>>> fn.name == 'plus'
True
>>> new_fn = fn.ch_attrs(func=minus)
>>> new_fn.func == minus
True
>>> new_fn.synopsis_string() == 'a,b -> plus -> sum'
True
>>>
>>>
>>> newer_fn = fn.ch_attrs(func=minus, name='sub', out='difference')
>>> newer_fn.synopsis_string() == 'a,b -> sub -> difference'
True
```

#### dot_lines(\*\*kwargs)

Returns a list of lines that can be used to make a dot graph

#### *classmethod* from_dict(dictionary)

The inverse of to_dict: Make a `FuncNode` from a dictionary of init args

#### *classmethod* has_as_instance(obj)

Verify if `obj` is an instance of a FuncNode (or specific sub-class).

The usefulness of this method is to not have to make a lambda with isinstance
when filtering.

```pycon
>>> FuncNode.has_as_instance(FuncNode(lambda x: x))
True
>>> FuncNode.has_as_instance("I am not a FuncNode: I'm a string")
False
```

#### names_maker(name=None, out=None)

This name maker will resolve names in the following fashion:

> 1. look at the (func) name and out given as arguments, if None…
> 2. use mk_func_name(func) to make names.

It will use the mk_func_name(func)  itself for out, but suffix the same with
an underscore to provide a mk_func_name.

This is so because here we want to allow easy construction of function networks
where a function’s output will be used as another’s input argument when
that argument has the the function’s (output) name.

#### node_validator()

Validates a func node. Raises ValidationError if something wrong. Returns None.

Validates:

* that the `func_node` params are valid, that is, if not `None`
  : * `func` should be a callable
    * `name` and `out` should be `str`
    * `bind` should be a `Dict[str, str]`
* that the names (`.name`, `.out` and all `.bind.values()`)
  : * are valid python identifiers (alphanumeric or underscore not starting with
      digit)
    * are not repeated (no duplicates)
* that `.bind.keys()` are indeed present as params of `.func`

#### synopsis_string(bind_info='values')

* **Parameters:**
  **bind_info** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'var_nodes'`, `'params'`, `'hybrid'`]) – 

  How to represent the bind in the synopsis string. Could be:
  - ’values’, `var_nodes` or `varnodes`: the values of the bind (default).
  - ’keys’ or ‘params’: the keys of the bind
  - ’hybrid’: the keys of the bind, but with the values that are the same as
    : the keys omitted.
* **Returns:**

```pycon
>>> fn = FuncNode(
...     func=lambda y, c: None , name='h', bind={'y': 'b', 'c': 'c'}, out='d'
... )
>>> fn.synopsis_string()
'b,c -> h -> d'
>>> fn.synopsis_string(bind_info='keys')
'y,c -> h -> d'
>>> fn.synopsis_string(bind_info='hybrid')
'y=b,c -> h -> d'
```

#### to_dict()

The inverse of from_dict: FuncNode.from_dict(fn.to_dict()) == fn

### *class* meshed.base.Mesh(func_nodes)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

### meshed.base.basic_node_validator(func_node)

Validates a func node. Raises ValidationError if something wrong. Returns None.

Validates:

* that the `func_node` params are valid, that is, if not `None`
  : * `func` should be a callable
    * `name` and `out` should be `str`
    * `bind` should be a `Dict[str, str]`
* that the names (`.name`, `.out` and all `.bind.values()`)
  : * are valid python identifiers (alphanumeric or underscore not starting with
      digit)
    * are not repeated (no duplicates)
* that `.bind.keys()` are indeed present as params of `.func`

### meshed.base.ch_func_node_attrs(fn, \*\*new_attrs_values)

Returns a copy of the func node with some of its attributes changed

```pycon
>>> def plus(a, b):
...     return a + b
...
>>> def minus(a, b):
...     return a - b
...
>>> fn = FuncNode(func=plus, out='sum')
>>> fn.func == plus
True
>>> fn.name == 'plus'
True
>>> new_fn = ch_func_node_attrs(fn, func=minus)
>>> new_fn.func == minus
True
>>> new_fn.synopsis_string() == 'a,b -> plus -> sum'
True
>>>
>>>
>>> newer_fn = ch_func_node_attrs(fn, func=minus, name='sub', out='difference')
>>> newer_fn.synopsis_string() == 'a,b -> sub -> difference'
True
```

### meshed.base.ensure_func_nodes(func_nodes)

Converts a list of objects to a list of FuncNodes.

### meshed.base.func_node_transformer(fn, kwargs_transformers=())

Get a modified `FuncNode` from an iterable of `kwargs_trans` modifiers.

### meshed.base.func_nodes_to_code(func_nodes, func_name='generated_pipeline', , favor_positional=True)

Convert an iterable of FuncNodes back to executable Python code.

This is the inverse operation of code_to_fnodes - it takes FuncNodes and generates
Python code that would create equivalent FuncNodes when parsed.
When favor_positional is True, any keyword argument with key equal to its value
is moved to the positional arguments list:

> func(a=a, b=b, c=z, d=d)  ->  func(a, b, c=z, d=d)
* **Parameters:**
  * **func_nodes** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`FuncNode`](#meshed.base.FuncNode)]) – Iterable of FuncNode instances to convert to code
  * **func_name** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Name for the generated function
  * **favor_positional** ([`bool`](https://docs.python.org/3/library/functions.html#bool)) – When True, transforms kwargs of the form key=key into positional args.
* **Return type:**
  [`str`](https://docs.python.org/3/library/stdtypes.html#str)
* **Returns:**
  String containing Python code

### meshed.base.get_init_params_of_instance(obj)

Get names of instance object `obj` that are also parameters of the
`__init__` of its class

### meshed.base.identifier_mapping(x)

Get an `IdentifierMapping` dict from a more loosely defined `Bind`.

You can get an identifier mapping (that is, an explicit for for a `bind` argument)
from…

… a single space-separated string

* **Return type:**
  [`dict`](https://docs.python.org/3/library/stdtypes.html#dict)[`Identifier` ([`str`](https://docs.python.org/3/library/stdtypes.html#str)), `Identifier` ([`str`](https://docs.python.org/3/library/stdtypes.html#str))]

```pycon
>>> identifier_mapping('x a_b yz')  #
{'x': 'x', 'a_b': 'a_b', 'yz': 'yz'}
```

… an iterable of strings or pairs of strings

```pycon
>>> identifier_mapping(['foo', ('bar', 'mitzvah')])
{'foo': 'foo', 'bar': 'mitzvah'}
```

… a dict will be considered to be the mapping itself

```pycon
>>> identifier_mapping({'x': 'y', 'a': 'b'})
{'x': 'y', 'a': 'b'}
```

### meshed.base.is_func_node(obj)

* **Return type:**
  [`bool`](https://docs.python.org/3/library/functions.html#bool)

```pycon
>>> is_func_node(FuncNode(lambda x: x))
True
>>> is_func_node("I am not a FuncNode: I'm a string")
False
```

### meshed.base.is_not_func_node(obj)

* **Return type:**
  [`bool`](https://docs.python.org/3/library/functions.html#bool)

```pycon
>>> is_not_func_node(FuncNode(lambda x: x))
False
>>> is_not_func_node("I am not a FuncNode: I'm a string")
True
```

### meshed.base.rebind_to_func(fnode, new_func)

Replaces `fnode.func` with `new_func`, changing the `.bind` accordingly.

```pycon
>>> fn = FuncNode(lambda x, y: x + y, bind={'x': 'X', 'y': 'Y'})
>>> fn.call_on_scope(dict(X=2, Y=3))
5
>>> new_fn = rebind_to_func(fn, lambda a, b, c=0: a * (b + c))
>>> new_fn.call_on_scope(dict(X=2, Y=3))
6
>>> new_fn.call_on_scope(dict(X=2, Y=3, c=1))
8
```

### meshed.base.underscore_func_node_names_maker(func, name=None, out=None)

This name maker will resolve names in the following fashion:

> 1. look at the (func) name and out given as arguments, if None…
> 2. use mk_func_name(func) to make names.

It will use the mk_func_name(func)  itself for out, but suffix the same with
an underscore to provide a mk_func_name.

This is so because here we want to allow easy construction of function networks
where a function’s output will be used as another’s input argument when
that argument has the the function’s (output) name.

### meshed.base.validate_that_func_node_names_are_sane(func_nodes)

Assert that the names of func_nodes are sane.
That is:

* are valid dot (graphviz) names (we’ll use str.isidentifier because lazy)
* All the `func.name` and `func.out` are unique
* more to come (TODO)…
