# meshed.scrap.collapse_and_expand

Ideas on collapsing and expanding nodes
See “Collapse and expand nodes” discussion:
[https://github.com/i2mint/meshed/discussions/54](https://github.com/i2mint/meshed/discussions/54)

### Functions

| [`collapse_function_calls`](#meshed.scrap.collapse_and_expand.collapse_function_calls)(src[, ...])               | Contract function calls in a source code string.                       |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`expand_function_calls`](#meshed.scrap.collapse_and_expand.expand_function_calls)(src[, call_func_name, ...]) | Inverse of collapse_function_calls.                                    |
| `expand_nodes`(dag[, nodes, is_node, ...])                                                         |                                                                        |
| `get_src_string`(src)                                                                              |                                                                        |
| [`remove_decorator_code`](#meshed.scrap.collapse_and_expand.remove_decorator_code)(src[, decorator_names])     | Remove the code corresponding to decorators from a source code string. |

### Classes

| [`CollapsedDAG`](#meshed.scrap.collapse_and_expand.CollapsedDAG)(dag)   | To collapse a DAG into a single function   |
|----------------------------------------------------------------------|--------------------------------------------|

### *class* meshed.scrap.collapse_and_expand.CollapsedDAG(dag)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

To collapse a DAG into a single function

This is useful for when you want to use a DAG as a function,
but you don’t want to see all the arguments.

### meshed.scrap.collapse_and_expand.collapse_function_calls(src, call_func_name='call', , rm_decorator='code_to_dag', include=None)

Contract function calls in a source code string.

That is, in source code, or a dag made from code_to_dag, replace calls of the form
`call(func, arg)` with `func(arg)`.

#### NOTE
Doesn’t work with arbitrary DAG src, only those made from code_to_dag.

### meshed.scrap.collapse_and_expand.expand_function_calls(src, call_func_name='call', , include=None)

Inverse of collapse_function_calls.
It replaces calls of the form `func(arg)` with `call(func, arg)`,
except when the function call is part of a function definition header.
If include is None, it expands all function calls.
If include is a list of function names, only those functions are expanded.
If include is a callable, it’s used as a filter function.

* **Return type:**
  [`str`](https://docs.python.org/3/library/stdtypes.html#str)

### meshed.scrap.collapse_and_expand.remove_decorator_code(src, decorator_names=None)

Remove the code corresponding to decorators from a source code string.
If decorator_names is None, will remove all decorators.
If decorator_names is an iterable of strings, will remove the decorators with those names.

* **Return type:**
  [`str`](https://docs.python.org/3/library/stdtypes.html#str)

### Examples

```pycon
>>> src = '''
... @decorator
... def func():
...     pass
... '''
>>> print(remove_decorator_code(src))
def func():
    pass
```

```pycon
>>> src = '''
... @decorator1
... @decorator2
... def func():
...     pass
... '''
>>> print(remove_decorator_code(src, "decorator1"))
@decorator2
def func():
    pass
```
