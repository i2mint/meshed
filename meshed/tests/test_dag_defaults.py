import pytest
from pytest import fixture
from meshed import DAG
from i2 import Sig


@fixture
def foo():
    def func(x, y=1):
        return x + y

    return func


def test_dag_with_defaults(foo):
    foo_dag = DAG([foo])
    assert foo_dag(0) == 1
    bar_dag = Sig(lambda x, y=2: None)(foo_dag)  # Bug: does not change dag.sig!!
    # bar_dag.sig = Sig(bar_dag)  # changed it manually
    assert str(Sig(bar_dag)) == "(x, y=2)"
    assert bar_dag(0) == 2  # Correct result after above change in dag._call
    assert True
