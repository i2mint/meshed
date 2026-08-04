"""Tests for ``meshed.dag.ch_funcs`` -- changing the functions of existing func nodes.

Besides the behaviour of ``ch_funcs`` itself, this module pins how ``ch_funcs`` may be
*called*: it is built with ``i2.double_up_as_factory``, so it must decorate when given
func nodes and only make a factory when not given any -- whether the func nodes are
passed positionally or by keyword.
"""

from functools import partial
from typing import NamedTuple

import meshed as ms
import pytest

import meshed.base
import meshed.util
from meshed.dag import DAG, ch_funcs, _validate_func_mapping
from meshed.tests.objects_for_testing import f, g
from meshed.base import compare_signatures
from i2 import Sig, double_up_as_factory


def _wrapped_by_keyword_is_supported() -> bool:
    """Whether the installed ``i2`` accepts a decorator's wrapped object by keyword.

    Before `i2mint/i2#82 <https://github.com/i2mint/i2/pull/82>`_,
    ``double_up_as_factory`` decided between "decorate this" and "make a factory" by
    looking only at the first *positional* argument.  A wrapped object passed by
    keyword therefore landed in ``**kwargs`` and the decorator silently returned a
    ``functools.partial`` factory instead of the decorated object.
    """

    @double_up_as_factory
    def probe(obj=None, *, unused=None):
        return obj

    return probe(obj=int) is int


#: Whether ``ch_funcs(func_nodes=...)`` decorates (True) or wrongly returns a factory
#: (False).  Depends on the installed ``i2`` -- see ``_wrapped_by_keyword_is_supported``.
I2_SUPPORTS_WRAPPED_BY_KEYWORD = _wrapped_by_keyword_is_supported()


@pytest.fixture
def example_func_nodes():

    funcs = [f, g]
    result = meshed.base._mk_func_nodes(funcs)
    return result


@pytest.fixture
def example_func_mapping():
    mapping = {"f_": f, "g_": g}
    return mapping


def test_ch_funcs_no_change(example_func_nodes):
    """Mapping every func node to the function it already has changes nothing."""
    funcs = [f, g]
    nodes = list(example_func_nodes)
    names = [node.name for node in nodes]

    dummy_mapping = dict(zip(names, funcs))

    new_dag = ch_funcs(nodes, func_mapping=dummy_mapping)

    assert isinstance(new_dag, DAG)
    assert nodes == new_dag.func_nodes


@pytest.mark.skipif(
    not I2_SUPPORTS_WRAPPED_BY_KEYWORD,
    reason=(
        "installed i2 predates i2mint/i2#82, so a double_up_as_factory decorator "
        "given its wrapped object by keyword wrongly returns a factory"
    ),
)
def test_ch_funcs_takes_func_nodes_by_keyword(example_func_nodes):
    """``ch_funcs(func_nodes=...)`` must decorate, not return a factory.

    Regression pin for `i2mint/i2#82 <https://github.com/i2mint/i2/pull/82>`_: passing
    the func nodes by keyword must mean exactly what passing them positionally means.
    Before that fix it returned a ``functools.partial``, and call sites papered over it
    with a trailing ``()`` -- which this test exists to stop coming back.
    """
    funcs = [f, g]
    nodes = list(example_func_nodes)
    dummy_mapping = dict(zip([node.name for node in nodes], funcs))

    new_dag = ch_funcs(func_nodes=nodes, func_mapping=dummy_mapping)

    assert not isinstance(new_dag, partial), "ch_funcs wrongly returned a factory"
    assert isinstance(new_dag, DAG)
    assert nodes == new_dag.func_nodes
    # ...and it agrees with the positional form
    assert new_dag.func_nodes == ch_funcs(nodes, func_mapping=dummy_mapping).func_nodes


class FlagWithMessage(NamedTuple):
    flag: bool
    msg: str = ""


# This function is used to give a more detailed report on
# mismatched signatures
# the same can be done by tweaking ch_func_node_func
# and its "alternative" param
def validate_func_mapping_on_signatures(func_mapping, func_nodes):
    """
    This function is used to give a more detailed report on
    mismatched signatures
    The same can be done by tweaking ch_func_node_func
    and its "alternative" param
    """
    from meshed import DAG

    _validate_func_mapping(func_mapping, func_nodes)
    d = dict()
    dag = DAG(func_nodes)
    for key, func in func_mapping.items():
        if fnode := dag._func_node_for.get(key, None):
            old_func = fnode.func

            if compare_signatures(old_func, func):
                result = FlagWithMessage(flag=True)
            else:
                msg = f"Signatures disagree for key={key}"
                result = FlagWithMessage(flag=False, msg=msg)

        else:
            msg = f"No funcnode matching the key {key}"
            result = FlagWithMessage(flag=False, msg=msg)
        d[key] = result
    all_flags_true = all(item.flag for item in d.values())
    return all_flags_true, d


def test_validate_func_mapping_based_on_signatures(
    example_func_nodes, example_func_mapping
):
    nodes = list(example_func_nodes)
    # funcs = [f, g]
    func_mapping = example_func_mapping
    result = validate_func_mapping_on_signatures(func_mapping, nodes)
    expected = (
        True,
        {
            "f_": FlagWithMessage(flag=True, msg=""),
            "g_": FlagWithMessage(flag=True, msg=""),
        },
    )
    assert result == expected


def test_validate_bind_attributes():
    """
    in ch_func_node_func: validate compatibility (not equality of sigs)
    we cannot use call_compatibility
    rename everything
    https://github.com/i2mint/i2/issues/47
    """
    pass
