"""Specific use of FuncNode and DAG"""

import re
from dataclasses import fields
from inspect import signature
from typing import Callable, Union, Iterable, Mapping
from functools import partial

from i2 import Sig, kwargs_trans
from meshed.base import FuncNode
from meshed.dag import DAG

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


def numbered_suffix_renamer(name, sep='_'):
    """
    >>> numbered_suffix_renamer('item')
    'item_1'
    >>> numbered_suffix_renamer('item_1')
    'item_2'
    """
    p = re.compile(sep + r'(\d+)$')
    m = p.search(name)
    if m is None:
        return f'{name}{sep}1'
    else:
        num = int(m.group(1)) + 1
        return p.sub(f'{sep}{num}', name)


Renamer = Union[Callable[[str], str], str, Mapping[str, str]]

# TODO: Extract  ingress/egress boilerplate to wrapper
def suffix_ids(
    func_nodes,
    renamer: Union[Renamer, str] = numbered_suffix_renamer,
    *,
    also_apply_to_func_label: bool = False,
):
    if isinstance(func_nodes, DAG):
        egress = DAG
    else:
        egress = list
    if isinstance(renamer, str):
        suffix = renamer
        renamer = lambda name: f'{name}{suffix}'
    assert callable(suffix), f'suffix needs to be callable'
    func_node_trans = func_node_name_trans(
        renamer, also_apply_to_func_label=also_apply_to_func_label,
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


def _rename_nodes(fn_kwargs, renamer: Renamer = numbered_suffix_renamer):
    fn_kwargs = fn_kwargs.copy()
    # decorate renamer so if the original returns None the decorated will return input
    renamer = _if_none_return_input(renamer)
    fn_kwargs['name'] = renamer(fn_kwargs['name'])
    fn_kwargs['out'] = renamer(fn_kwargs['out'])
    fn_kwargs['bind'] = {
        param: renamer(var_id) for param, var_id in fn_kwargs['bind'].items()
    }
    return fn_kwargs


# TODO: Postelize? Work with func_nodes or dag?
# TODO: Extract ingress/egress boilerplate to wrapper
def rename_nodes(func_nodes: DagAble, renamer: Renamer = numbered_suffix_renamer):
    """Renames variables and functions of a ``DAG`` or iterable of ``FuncNodes``.

    :param func_nodes: A ``DAG`` of iterable of ``FuncNodes``
    :param renamer: A function taking an old name and returning the new one, or:
        - A dictionary ``{old_name: new_name, ...}`` mapping old names to new ones
        - A string, which will be appended to all identifiers of the ``func_nodes``
    :return: func_nodes with some or all identifiers changed. If the input ``func_nodes``
    is an iterable of ``FuncNodes``, a list of func_nodes will be returned, and if the
    input ``func_nodes`` is a ``DAG`` instance, a ``DAG`` will be returned.

    >>> from meshed.makers import code_to_dag
    >>> from meshed.dag import print_dag_string
    >>>
    >>> @code_to_dag
    ... def dag():
    ...     b = f(a)
    ...     c = g(x=a)
    ...     d = h(b, y=c)
    ...

    This is what the dag looks like:

    >>> print_dag_string(dag)
    x=a -> g -> c
    a -> f -> b
    b,y=c -> h -> d

    Now, if rename the vars of the ``dag`` without further specifying how, all of our
    nodes (names) will be suffixed with a ``_1``

    >>> new_dag = rename_nodes(dag)
    >>> print_dag_string(new_dag)
    a=a_1 -> f_1 -> b_1
    x=a_1 -> g_1 -> c_1
    b=b_1,y=c_1 -> h_1 -> d_1

    If any nodes are already suffixed by ``_`` followed by a number, the default
    renamer (``numbered_suffix_renamer``) will increment that number:

    >>> another_new_data = rename_nodes(new_dag)
    >>> print_dag_string(another_new_data)
    x=a_2 -> g_2 -> c_2
    a=a_2 -> f_2 -> b_2
    b=b_2,y=c_2 -> h_2 -> d_2

    If we specify a string for the ``renamer`` argument, it will be used to suffix all
    the nodes.

    >>> print_dag_string(rename_nodes(dag, renamer='_copy'))
    a=a_copy -> f_copy -> b_copy
    x=a_copy -> g_copy -> c_copy
    b=b_copy,y=c_copy -> h_copy -> d_copy

    Finally, for full functionality on renaming, you can use a function

    >>> print_dag_string(rename_nodes(dag, renamer=lambda x: f"{x.upper()}"))
    a=A -> F -> B
    x=A -> G -> C
    b=B,y=C -> H -> D

    In all the above our input was a ``DAG`` so we got a ``DAG`` back, but if we enter
    an iterable of ``FuncNode`` instances, we'll get a list of the same back.
    Also, know that if your function returns ``None`` for a given identifier, it will
    have the effect of not changing that identifier.

    >>> rename_nodes(dag.func_nodes, renamer=lambda x: x.upper() if x in 'abc' else None)
    [FuncNode(x=A -> g -> C), FuncNode(a=A -> f -> B), FuncNode(b=B,y=C -> h -> d)]

    If you want to rename the nodes with an explicit mapping, you can do so by
    specifying this mapping as your renamer

    >>> substitutions = {'a': 'alpha', 'b': 'bravo'}
    >>> print_dag_string(rename_nodes(dag, renamer=substitutions))
    a=alpha -> f -> bravo
    x=alpha -> g -> c
    b=bravo,y=c -> h -> d

    """
    if isinstance(func_nodes, DAG):
        egress = DAG
    else:
        egress = list
    if isinstance(renamer, str):
        suffix = renamer
        renamer = lambda name: f'{name}{suffix}'
    elif isinstance(renamer, Mapping):
        old_to_new_map = dict(renamer)
        renamer = old_to_new_map.get
    assert callable(renamer), f'Could not be resolved into a callable: {renamer}'
    ktrans = partial(_rename_nodes, renamer=renamer)
    func_node_trans = partial(func_node_transformer, kwargs_transformers=ktrans)
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
