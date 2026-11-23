"""Regression tests for meshed#104: ``suffix_ids`` with a callable renamer, and ``ReactiveScope``."""

from meshed import DAG, FuncNode
from meshed.composition import suffix_ids
from meshed.scrap.reactive_scope import ReactiveScope


def ff(a):
    return a


def test_suffix_ids_with_the_default_callable_renamer():
    (node,) = suffix_ids([FuncNode(ff, out="x")])
    assert node.name == "ff_1"


def test_suffix_ids_with_a_suffix_string():
    (node,) = suffix_ids([FuncNode(ff, out="x")], "_s")
    assert node.name == "ff_s"


def test_suffix_ids_returns_a_dag_given_a_dag():
    assert isinstance(suffix_ids(DAG([FuncNode(ff, out="x")])), DAG)


def test_reactive_scope_computes_when_inputs_are_written():
    def add(a, b):
        return a + b

    s = ReactiveScope([FuncNode(add, out="c")])
    s["a"] = 1
    assert "c" not in s
    s["b"] = 2
    assert s["c"] == 3


import pytest


@pytest.mark.parametrize(
    "renamer, expected",
    [(lambda x: x + "_X", "ff_X"), (str.upper, "FF")],
)
def test_suffix_ids_with_a_renamer_of_any_parameter_name(renamer, expected):
    """A callable renamer need not name its parameter ``name`` (issue #111)."""
    from meshed import FuncNode
    from meshed.composition import suffix_ids

    def ff(x):
        return x

    (node,) = suffix_ids([FuncNode(ff, name="ff", out="y")], renamer=renamer)
    assert node.name == expected
