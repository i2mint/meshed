"""``name=`` alias of ``FuncNode.func_id`` (meshed#20) and the id-keyed ``DAG.graph`` (meshed#86)."""

import dataclasses
from dataclasses import dataclass

from meshed import DAG, FuncNode


def test_replace_with_name():
    f = FuncNode(lambda x: x, name="a", out="o")
    assert dataclasses.replace(f, name="b").func_id == "b"


def test_name_on_dataclass_subclass():
    @dataclass
    class Sub(FuncNode):
        extra: int = 0

    s = Sub(lambda x: x, name="q", extra=2)
    assert (s.func_id, s.extra) == ("q", 2)


def test_graph_keyed_by_ids():
    def add(a, b=1):
        return a + b

    dag = DAG([add])
    assert dag.graph == {"a": ["add_"], "b": ["add_"], "add_": ["add"]}
    assert all(isinstance(k, str) for k in dag.graph)
