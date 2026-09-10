# meshed.composition

Specific use of FuncNode and DAG

### Functions

| `func_node_kwargs_trans`(func)                                                               |                                                                             |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`func_node_name_trans`](#meshed.composition.func_node_name_trans)(name_trans, \*[, ...]) |                                                                             |
| [`get_param`](#meshed.composition.get_param)(func)                             | Find the name of the parameter of a function with exactly one parameter.    |
| [`is_func_node_kwargs_trans`](#meshed.composition.is_func_node_kwargs_trans)(func)             | Returns True iff the only required params of func are FuncNode field names. |
| [`line_with_dag`](#meshed.composition.line_with_dag)(\*steps)                      | Emulate a Line object with a DAG                                            |
| `suffix_ids`(func_nodes[, renamer, ...])                                                     |                                                                             |

### meshed.composition.func_node_name_trans(name_trans, , also_apply_to_func_label=False)

* **Parameters:**
  * **name_trans** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/library/stdtypes.html#str)], [`str`](https://docs.python.org/3/library/stdtypes.html#str) | [`None`](https://docs.python.org/3/library/constants.html#None)]) – A function taking a str and returning a str, or None (to indicate
    that no transformation should take place).
  * **also_apply_to_func_label** ([`bool`](https://docs.python.org/3/library/functions.html#bool))
* **Returns:**

### meshed.composition.get_param(func)

Find the name of the parameter of a function with exactly one parameter.
Raise an error if more or less parameters.

* **Parameters:**
  **func** – callable, the function to inspect
* **Returns:**
  str, the name of the single parameter of func

### meshed.composition.is_func_node_kwargs_trans(func)

Returns True iff the only required params of func are FuncNode field names.
This ensures that the func will be able to be bound to FuncNode fields and
therefore used as a func_node (kwargs) transformer.

* **Return type:**
  [`bool`](https://docs.python.org/3/library/functions.html#bool)

### meshed.composition.line_with_dag(\*steps)

Emulate a Line object with a DAG

* **Parameters:**
  **steps** – an iterable of callables, the steps of the pipeline. Each step should have exactly one parameter
  and the output of each step is fed into the next
* **Returns:**
  a DAG instance computing the composition of all the functions in steps, in the provided order
