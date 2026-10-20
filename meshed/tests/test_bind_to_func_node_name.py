"""Binding a parameter to a func node's name means that node's output (#19)."""

import pytest
from i2 import Sig

from meshed import DAG, FuncNode, BindToFuncNodeNameWarning
from meshed.util import ValidationError


def foo(a, b):
    return a + b


def bar(x, y):
    return x * y


def test_bind_to_func_node_name_means_its_output():
    with pytest.warns(BindToFuncNodeNameWarning, match="y='foo' -> '_foo'"):
        dag = DAG([FuncNode(foo, name="foo"), FuncNode(bar, bind={"y": "foo"})])
    assert str(Sig(dag)) == "(a, b, x)"
    assert dag(1, 2, 3) == 9


def test_bind_to_func_node_name_in_reverse_order():
    with pytest.warns(BindToFuncNodeNameWarning):
        dag = DAG([FuncNode(bar, bind={"y": "foo"}), FuncNode(foo, name="foo")])
    assert dag(a=1, b=2, x=3) == 9


def test_var_node_wins_over_func_node_name():
    # 'y' is both the name of g's node and the out of f's node: it's the var node
    def f(a):
        return a + 1

    def g(b):
        return b * 10

    def h(p):
        return p

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", BindToFuncNodeNameWarning)
        dag = DAG(
            [
                FuncNode(f, name="f", out="y"),
                FuncNode(g, name="y", out="z"),
                FuncNode(h, bind={"p": "y"}),
            ]
        )
    assert dag(a=1, b=2) == (20, 2)


def test_bind_to_func_node_out_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", BindToFuncNodeNameWarning)
        dag = DAG([FuncNode(foo, name="foo"), FuncNode(bar, bind={"y": "_foo"})])
    assert dag(1, 2, 3) == 9


def test_bind_to_function_means_its_node_out():
    # i2mint/meshed#96
    dag = DAG([FuncNode(foo), FuncNode(bar, bind={"y": foo})])
    assert str(Sig(dag)) == "(a, b, x)"
    assert dag(1, 2, 3) == 9
    # order does not matter, nor does a custom out
    dag = DAG([FuncNode(bar, bind={"y": foo}), FuncNode(foo, out="s")])
    assert dag(a=1, b=2, x=3) == 9


def test_bind_to_function_before_dag_shows_in_synopsis():
    assert str(FuncNode(bar, bind={"y": foo})) == "FuncNode(y=<foo>,x -> bar_ -> bar)"


def test_bind_to_function_not_in_dag_raises():
    def baz(z):
        return z

    with pytest.raises(ValidationError, match="no func node"):
        DAG([FuncNode(baz), FuncNode(bar, bind={"y": foo})])


def test_bind_to_function_wrapped_twice_raises():
    with pytest.raises(ValidationError, match="several func nodes"):
        DAG(
            [
                FuncNode(foo, out="s"),
                FuncNode(foo, name="foo2", out="t"),
                FuncNode(bar, bind={"y": foo}),
            ]
        )


def test_bind_to_function_with_variadics():
    def total(*nums):
        return sum(nums)

    dag = DAG([FuncNode(total), FuncNode(bar, bind={"y": total})])
    assert dag(nums=(1, 2), x=3) == 9


def test_bind_value_neither_str_nor_callable_raises():
    with pytest.raises(ValidationError, match="Should be a str"):
        FuncNode(bar, bind={"y": 3})
