# meshed.scrap.annotations_to_meshes

Code related to work on the “From annotated functions to meshes” discussion:

[https://github.com/i2mint/meshed/discussions/55](https://github.com/i2mint/meshed/discussions/55)

### Functions

| [`callable_annots_to_signature`](#meshed.scrap.annotations_to_meshes.callable_annots_to_signature)(callable_annots)   | Produces a signature from a Callable type annotation                                             |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| [`func_types_to_protocol`](#meshed.scrap.annotations_to_meshes.func_types_to_protocol)(func_types[, name, ...]) | Produces a typing.Protocol based on a dictionary of `(method_name, Callable_type)` specification |
| [`func_types_to_scaffold`](#meshed.scrap.annotations_to_meshes.func_types_to_scaffold)(func_types[, name])      | Produces a scaffold class containing the said methods, with given annotations                    |
| `test_func_types_to_protocol`()                                                                  |                                                                                                  |
| `test_func_types_to_scaffold`()                                                                  |                                                                                                  |
| `try_annotation_name`(arg_annotation, ...)                                                       |                                                                                                  |

### meshed.scrap.annotations_to_meshes.callable_annots_to_signature(callable_annots, mk_argname=<function try_annotation_name>)

Produces a signature from a Callable type annotation

* **Return type:**
  [`Signature`](https://docs.python.org/3/library/inspect.html#inspect.Signature)

```pycon
>>> from typing import Callable, NewType
>>> MyType = NewType('MyType', str)
>>> sig = callable_annots_to_signature(Callable[[MyType, str], str])
>>> import inspect
>>> isinstance(sig, inspect.Signature)
True
>>> list(sig.parameters.keys())
['self', 'mytype', 'arg_01']
>>> sig.parameters['arg_01'].annotation
<class 'str'>
```

### meshed.scrap.annotations_to_meshes.func_types_to_protocol(func_types, name=None, \*, mk_argname=<function try_annotation_name>)

Produces a typing.Protocol based on a dictionary of
`(method_name, Callable_type)` specification

* **Return type:**
  [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

### meshed.scrap.annotations_to_meshes.func_types_to_scaffold(func_types, name=None)

Produces a scaffold class containing the said methods, with given annotations

* **Return type:**
  [`str`](https://docs.python.org/3/library/stdtypes.html#str)
