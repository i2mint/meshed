"""
Code related to work on the "From annotated functions to meshes" discussion:

https://github.com/i2mint/meshed/discussions/55


"""

import typing
from typing import Dict, Protocol, Callable, TypeVar, Any
from collections.abc import Callable as CallableGenericAlias
from inspect import signature, Signature, Parameter
from functools import wraps

import builtins
import re

_camel_pattern = re.compile(r"(?<!^)(?=[A-Z])")
PK = Parameter.POSITIONAL_OR_KEYWORD


def _camel_to_snake(x):
    return _camel_pattern.sub("_", x)


_is_lower = lambda x: x == x.lower()
_builtin_lower_names = set(map(_camel_to_snake, filter(_is_lower, dir(builtins))))


Annotation, ArgPosition, MethodName, Argname = Any, int, str, str
MkArgname = Callable[[Annotation, ArgPosition, MethodName], Argname]


def try_annotation_name(
    arg_annotation: Annotation, arg_position: ArgPosition, method_name: MethodName
) -> Argname:
    argname = getattr(arg_annotation, "__name__", None)
    if argname is None or argname in _builtin_lower_names:
        argname = f"arg_{arg_position:02.0f}"
    return argname.lower()


def _is_callable_type_annot(x):
    return isinstance(x, CallableGenericAlias)


def callable_annots_to_signature(
    callable_annots: CallableGenericAlias, mk_argname: MkArgname = try_annotation_name
) -> Signature:
    """Produces a signature from a Callable type annotation

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

    """
    origin = typing.get_origin(callable_annots)
    if not _is_callable_type_annot(origin):
        raise ValueError("The provided type is not a Callable generic alias")

    input_annots, return_annot = typing.get_args(callable_annots)
    arg_params = [
        Parameter(mk_argname(annot, i, None), PK, annotation=annot)
        for i, annot in enumerate(input_annots)
    ]
    self_param = Parameter("self", PK)
    return Signature([self_param] + arg_params, return_annotation=return_annot)


def func_types_to_protocol(
    func_types: Dict[str, CallableGenericAlias],
    name: str = None,
    *,
    mk_argname: MkArgname = try_annotation_name,
) -> typing.Protocol:
    """Produces a typing.Protocol based on a dictionary of
    `(method_name, Callable_type)` specification"""

    class TempProtocol(Protocol):
        pass

    if name:
        TempProtocol.__name__ = name

    for method_name, callable_type in func_types.items():
        sig = callable_annots_to_signature(callable_type, mk_argname)

        def method_stub(*args, **kwargs):
            pass

        method_stub.__signature__ = sig
        setattr(TempProtocol, method_name, method_stub)

    return TempProtocol


def test_func_types_to_protocol():
    from typing import Iterable, Any, NewType, Callable

    Group = NewType("Group", str)
    Item = NewType("Item", Any)

    class Groups:
        add_item_to_group: Callable[[Item, Group], Any]
        add_items_to_group: Callable[[Iterable[Item], Group], Any]
        list_groups: Callable[[], Iterable[Group]]
        items_for_group: Callable[[Group], Iterable[Item]]

    class ExpectedGroupsProtocol(Protocol):
        def add_items_to_group(self, item: Item, group: Group) -> Any:
            pass

        def add_items_to_group(self, items: Iterable[Item], group: Group) -> Any:
            pass

        def list_groups(self) -> Iterable[Group]:
            pass

        def items_for_group(self, group: Group) -> Iterable[Item]:
            pass

    ActualGroupsProtocol = func_types_to_protocol(Groups.__annotations__)
    # TODO: assert that ActualGroupsProtocol and ExpectedGroupsProtocol are equal in
    #  some way (but == doesn't work, and is not meant to!)


def func_types_to_scaffold(
    func_types: Dict[str, CallableGenericAlias], name: str = None
) -> str:
    """Produces a scaffold class containing the said methods, with given annotations"""

    if name is None:
        name = "GeneratedClass"

    methods = []
    for method_name, callable_type in func_types.items():
        sig = callable_annots_to_signature(callable_type)
        arg_str = ", ".join(
            (
                f"{param.name}: {param.annotation.__name__}"
                if param.annotation != Parameter.empty
                else f"{param.name}"
            )
            for param in sig.parameters.values()
        )
        return_annotation = (
            sig.return_annotation.__name__
            if sig.return_annotation != Parameter.empty
            else "None"
        )
        method_str = (
            f"def {method_name}({arg_str}) -> {return_annotation}:\n    \tpass\n"
        )
        methods.append(method_str)

    class_str = f"\nclass {name}:\n    " + "\n    ".join(methods)
    return class_str


_expected_scaffold = """
class GeneratedClass:
    def add_item_to_group(self, item: Item, group: Group) -> Any:
    	pass

    def add_items_to_group(self, iterable: Iterable, group: Group) -> Any:
    	pass

    def list_groups(self) -> Iterable:
    	pass

    def items_for_group(self, group: Group) -> Iterable:
    	pass
"""


def test_func_types_to_scaffold():
    from typing import Iterable, Any, NewType, Callable

    Group = NewType("Group", str)
    Item = NewType("Item", Any)

    class Groups:
        add_item_to_group: Callable[[Item, Group], Any]
        add_items_to_group: Callable[[Iterable[Item], Group], Any]
        list_groups: Callable[[], Iterable[Group]]
        items_for_group: Callable[[Group], Iterable[Item]]

    actual_scaffold = func_types_to_scaffold(Groups.__annotations__)
    assert actual_scaffold == _expected_scaffold
