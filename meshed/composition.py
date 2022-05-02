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
    therefore used as a func_node (kwargs) transformer.
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
    """

    :param name_trans: A function taking a str and returning a str, or None (to indicate
        that no transformation should take place).
    :param also_apply_to_func_label:
    :return:
    """
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


def _add_suffix(x, suffix):
    return f"{x}{suffix}"


_incremental_suffixes = _suffix()
_renamers = (lambda x: f'{x}{suffix}' for suffix in _incremental_suffixes)

# TODO: Postelize? Work with func_nodes or dag?
# TODO: Extract  ingress/egress boilerplate to wrapper
def suffix_ids(
    func_nodes, suffix: Optional[str] = None, *, also_apply_to_func_label: bool = False
):
    if isinstance(func_nodes, DAG):
        egress = DAG
    else:
        egress = list
    if suffix is None:
        suffix = next(_incremental_suffixes)
    func_node_trans = func_node_name_trans(
        lambda name: f'{name}{suffix}',
        also_apply_to_func_label=also_apply_to_func_label,
    )
    return egress(map(func_node_trans, func_nodes))


def _if_none_return_input(func):
    """Wraps a function so that when the original func outputs None, the wrapped will
    return the original input instead.

    >>> def func(x):
    ...     if x % 2 == 0:
    ...         return None
    ...     else:
    ...         return x * 10
    >>> wfunc = _if_none_return_input(func)
    >>> func(3)
    30
    >>> wfunc(3)
    30
    >>> assert func(4) is None
    >>> wfunc(4)
    4
    """
    def _func(input_val):
        if (output_val := func(input_val)) is not None:
            return output_val
        else:
            return input_val

    return _func


def _rename_vars(fn_kwargs, renamer=None):
    if renamer is None:
        renamer = next(_renamers)
    fn_kwargs = fn_kwargs.copy()
    renamer = _if_none_return_input(renamer)  # if renamer returns None, return input
    fn_kwargs['name'] = renamer(fn_kwargs['name'])
    fn_kwargs['out'] = renamer(fn_kwargs['out'])
    fn_kwargs['bind'] = {
        param: renamer(var_id) for param, var_id in fn_kwargs['bind'].items()
    }
    return fn_kwargs


# TODO: Postelize? Work with func_nodes or dag?
# TODO: Extract ingress/egress boilerplate to wrapper
def rename_vars(func_nodes: DagAble, renamer=None):
    """Renames variables and functions of a ``DAG`` or iterable of ``FuncNodes``.

    :param func_nodes: A ``DAG`` of iterable of ``FuncNodes``
    :param renamer:
    :return:

    >>> from meshed.makers import code_to_dag
    >>>
    >>> @code_to_dag
    ... def dag():
    ...     b = f(a)
    ...     c = g(x=a)
    ...     d = h(b, y=c)
    ...
    >>> print(dag.synopsis_string(bind_info='hybrid'))
    x=a -> g -> c
    a -> f -> b
    b,y=c -> h -> d
    >>> print(rename_vars(dag, renamer='_2').synopsis_string(bind_info='hybrid'))
    a=a_2 -> f_2 -> b_2
    x=a_2 -> g_2 -> c_2
    b=b_2,y=c_2 -> h_2 -> d_2
    >>> new_func_nodes = rename_vars(dag.func_nodes, renamer=lambda x: f"{x}_2")
    >>> [fn.synopsis_string(bind_info='hybrid') for fn in new_func_nodes]
    ['x=a_2 -> g_2 -> c_2', 'a=a_2 -> f_2 -> b_2', 'b=b_2,y=c_2 -> h_2 -> d_2']
    """
    if isinstance(func_nodes, DAG):
        egress = DAG
    else:
        egress = list
    if renamer is None:
        renamer = next(_incremental_suffixes)
    if isinstance(renamer, str):
        renamer = partial(_add_suffix, suffix=renamer)

    ktrans = partial(_rename_vars, renamer=renamer)
    func_node_trans = partial(
        func_node_transformer,
        kwargs_transformers=ktrans
    )
    return egress(map(func_node_trans, func_nodes))


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
