"""Tests for the ``RepeatedNamesWarning`` a ``DAG`` raises when func nodes share a name
(see i2mint/meshed#17)."""

import warnings

import pytest

from meshed import DAG, FuncNode, RepeatedNamesWarning, repeated_names


def _inc(x):
    return x + 1


def test_reusing_a_function_warns_and_still_works():
    nodes = [
        FuncNode(_inc, out="a", bind={"x": "x"}),
        FuncNode(_inc, out="b", bind={"x": "a"}),
    ]
    with pytest.warns(RepeatedNamesWarning, match="_inc"):
        dag = DAG(nodes)
    assert dag(1) == 3
    assert [n.name for n in dag.func_nodes] == ["_inc", "_inc__2"]


def test_explicit_names_do_not_warn():
    nodes = [
        FuncNode(_inc, name="first", out="a", bind={"x": "x"}),
        FuncNode(_inc, name="second", out="b", bind={"x": "a"}),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RepeatedNamesWarning)
        assert DAG(nodes)(1) == 3


def test_repeated_names():
    assert repeated_names([FuncNode(_inc, out="a"), FuncNode(_inc, out="b")]) == {
        "_inc": 2
    }
