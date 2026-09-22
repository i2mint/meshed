"""Tests for the ``DuplicateOutsWarning`` a ``DAG`` raises when two func nodes write
to the same var node (see i2mint/meshed#40)."""

import warnings

import pytest

from meshed import DAG, FuncNode
from meshed.dag import DuplicateOutsWarning


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
