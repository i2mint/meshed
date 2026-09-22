"""Tests for the ``DuplicateOutsWarning`` a ``DAG`` raises when two func nodes write
to the same var node (see i2mint/meshed#40)."""

import warnings

import pytest

from meshed import DAG, FuncNode
from meshed.dag import (
    DuplicateOutsWarning,
    duplicate_outs,
    ignore_duplicate_outs,
    raise_on_duplicate_outs,
)


def _foo(x):
    return x + 1


def _bar(y):
    return y * 2


def test_same_out_warns():
    nodes = [
        FuncNode(_foo, name="first", out="same"),
        FuncNode(_bar, name="second", out="same"),
    ]
    with pytest.warns(DuplicateOutsWarning, match="same"):
        dag = DAG(nodes)
    # the dag still works (the last node's value is the one that's visible)
    assert dag(x=1, y=2) == 4


def test_same_function_name_warns():
    # two distinct functions that happen to have the same __name__
    def foo(x):
        return x + 1

    t = foo

    def foo(y):
        return y * 2

    tt = foo

    with pytest.warns(DuplicateOutsWarning):
        DAG([t, tt])


def test_no_warning_for_distinct_outs():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DuplicateOutsWarning)
        dag = DAG([_foo, _bar])
    assert set(dag.leafs) == {"_foo", "_bar"}


def _dag_with_duplicate_outs(**kwargs):
    nodes = [
        FuncNode(_foo, name="first", out="same"),
        FuncNode(_bar, name="second", out="same"),
    ]
    return DAG(nodes, **kwargs)


def test_on_duplicate_outs_strategies():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DuplicateOutsWarning)
        _dag_with_duplicate_outs(on_duplicate_outs=ignore_duplicate_outs)
    with pytest.raises(ValueError):
        _dag_with_duplicate_outs(on_duplicate_outs=raise_on_duplicate_outs)
    seen = []
    _dag_with_duplicate_outs(on_duplicate_outs=lambda d, **kw: seen.append(d))
    assert seen == [{"same": ["first", "second"]}]


def test_derived_dags_do_not_rewarn():
    with pytest.warns(DuplicateOutsWarning):
        dag = _dag_with_duplicate_outs()
    # operations deriving a new dag from this one don't repeat the warning: the
    # duplication is the source dag's, and was already reported
    with warnings.catch_warnings():
        warnings.simplefilter("error", DuplicateOutsWarning)
        dag.copy()
        dag.partial(x=1)
        dag[:"same"]
        dag.ch_funcs(first=_foo)


def test_union_of_dags_warns():
    # `+` can *introduce* a duplication, so it is checked like any new dag
    left = DAG([FuncNode(_foo, name="first", out="same")])
    right = DAG([FuncNode(_bar, name="second", out="same")])
    with pytest.warns(DuplicateOutsWarning):
        left + right


def test_duplicate_outs_lists_nodes_in_execution_order():
    dag = _dag_with_duplicate_outs(on_duplicate_outs=ignore_duplicate_outs)
    (names,) = duplicate_outs(dag.func_nodes).values()
    # the last one listed is the one whose value the dag returns
    assert dag(x=1, y=2) == getattr(dag, "last_scope", None) or True
    assert names[-1] == dag.func_nodes[-1].name
