"""``i2``'s ``NotSet`` sentinel in a signature means "required / no default".

See i2mint/i2#48: once ``i2.FuncFactory`` shows ``NotSet`` defaults, a DAG holding
such a node must keep the signature it has today. Otherwise its positional order
changes silently (``NotSet``-defaulted params would be sorted after required ones),
and merging with a same-named, non-defaulted param would raise.
These tests use ``i2.deco.NotSet`` directly, so they pass with any i2 version.
"""

import pytest
from i2 import Sig
from i2.deco import NotSet

from meshed import DAG


def g(a):
    return a


# ``g`` with a ``NotSet`` default, as a re-landed i2#88 ``FuncFactory`` would show.
def g_with_not_set(a=NotSet):
    return a


g_with_not_set.__name__ = g.__name__


def h(z):
    return z * 10


def k(a, z):
    return a - z


def test_dag_signature_is_the_same_as_without_not_set():
    dag, plain = DAG([g_with_not_set, h]), DAG([g, h])
    assert str(Sig(dag)) == str(Sig(plain)) == "(a, z)"
    assert dag(1, 2) == plain(1, 2)  # positional order is kept


def test_not_set_param_merges_with_same_named_required_param():
    dag = DAG([g_with_not_set, k])  # used to raise: "didn't have the same default"
    assert str(Sig(dag)) == str(Sig(DAG([g, k])))


def test_not_set_root_is_required():
    with pytest.raises(TypeError):
        DAG([g_with_not_set])()


def test_real_defaults_are_kept():
    def f(a=3):
        return a

    assert DAG([f])() == 3
