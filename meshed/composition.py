"""Specific use of FuncNode and DAG"""

from dataclasses import fields
from inspect import signature
from typing import Callable, Union, Iterable, Optional
from functools import partial

from meshed import FuncNode, DAG
from i2 import Sig, kwargs_trans

_func_node_fields = {x.name for x in fields(FuncNode)}

# TODO: How to enforce that it only has keys from _func_node_fields?
FuncNodeKwargs = dict
FuncNodeKwargsTrans = Callable[[FuncNodeKwargs], FuncNodeKwargs]
FuncNodeAble = Union[FuncNode, Callable]
DagAble = Union[DAG, Iterable[FuncNodeAble]]


class InvalidFunctionParameters(ValueError):
    """To be used when a function's parameters are not compliant with some rule about
    them."""


def func_node_transformer(
    fn: FuncNode, kwargs_transformers=(),
):
    """Get a modified ``FuncNode`` from an iterable of ``kwargs_trans`` modifiers."""
    func_node_kwargs = fn.to_dict()
    if callable(kwargs_transformers):
        kwargs_transformers = [kwargs_transformers]
    for trans in kwargs_transformers:
        if (new_kwargs := trans(func_node_kwargs)) is not None:
            func_node_kwargs = new_kwargs
    return FuncNode.from_dict(func_node_kwargs)


def is_func_node_kwargs_trans(func: Callable) -> bool:
    """Returns True iff the only required params of func are FuncNode field names.
    This ensures that the func will be able to be bound to FuncNode fields and
    therefore used
    as a func_node (kwargs) transformer.
    """
    return _func_node_fields.issuperset(Sig(func).required_names)


def func_node_kwargs_trans(func: Callable) -> FuncNodeKwargsTrans:
    if is_func_node_kwargs_trans(func):
        return func
    else:
        raise InvalidFunctionParameters(
            f'A FuncNodeKwargsTrans is expected to only have required params that are '
            f'also '
            f"FuncNode fields ({', '.join(_func_node_fields)}). "
            'Function {func} had signature: {Sig(func)}'
        )


def func_node_name_trans(
    name_trans: Callable[[str], Union[str, None]],
    *,
    also_apply_to_func_label: bool = False,
):
    sig = Sig(name_trans)
    if not is_func_node_kwargs_trans(name_trans):
        if Sig(name_trans).n_required <= 1:

            def name_trans(name):
                return name_trans(name)

    kwargs_trans_kwargs = dict(name=name_trans)
    if also_apply_to_func_label:
        kwargs_trans_kwargs.update(func_label=name_trans)

    return partial(
        func_node_transformer,
        kwargs_transformers=partial(kwargs_trans, **kwargs_trans_kwargs),
    )


def _suffix():
    i = 0
    while True:
        yield f'_{i}'
        i += 1


# TODO: Postelize? Work with func_nodes or dag?
def suffix_ids(
    func_nodes, suffix: Optional[str] = None, *, also_apply_to_func_label: bool = False
):
    if suffix is None:
        suffix = next(_suffix)
    func_node_trans = func_node_name_trans(
        lambda name: f'{name}{suffix}',
        also_apply_to_func_label=also_apply_to_func_label,
    )
    return list(map(func_node_trans, func_nodes))


# ---------------------------------------------------------------------------------------
# lined, with meshed


def get_param(func):
    """
    Find the name of the parameter of a function with exactly one parameter.
    Raise an error if more or less parameters.
    :param func: callable, the function to inspect
    :return: str, the name of the single parameter of func
    """

    params = signature(func).parameters.keys()
    assert (
        len(params) == 1
    ), f"Your function has more than 1 parameter! Namely: {', '.join(params)}"
    for param in params:
        return param


def line_with_dag(*steps):
    """
    Emulate a Line object with a DAG
    :param steps: an iterable of callables, the steps of the pipeline. Each step should have exactly one parameter
    and the output of each step is fed into the next
    :return: a DAG instance computing the composition of all the functions in steps, in the provided order
    """

    step_counter = 0
    first_node = FuncNode(steps[0], out=f'step_{step_counter}')
    func_nodes = [first_node]
    for step in steps[1:]:
        step_node = FuncNode(
            step,
            out=f'step_{step_counter + 1}',
            bind={get_param(step): f'step_{step_counter}'},
        )
        step_counter += 1
        func_nodes.append(step_node)

    return DAG(func_nodes)
