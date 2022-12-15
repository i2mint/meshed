import meshed as ms
import pytest

import meshed.base
import meshed.util
from meshed.dag import ch_funcs, _validate_func_mapping
from meshed.tests.objects_for_testing import f, g
from meshed.base import compare_signatures
from i2 import Sig
from typing import NamedTuple


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
    funcs = [f, g]
    nodes = list(example_func_nodes)
    names = [node.name for node in nodes]

    dummy_mapping = dict(zip(names, funcs))

    new_dag = ch_funcs(
        func_nodes=nodes,
        func_mapping=dummy_mapping,
    )
    new_nodes = new_dag().func_nodes
    assert nodes == new_nodes


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
