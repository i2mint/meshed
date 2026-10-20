"""Moved: :class:`CachedDag` now lives in :mod:`meshed.cached_dag` (i2mint/meshed#36).

This module re-exports it for code that imported it from ``meshed.scrap``.
"""

from meshed.cached_dag import (  # noqa: F401
    CachedDag,
    NoOverwritesDict,
    NotAllowed,
    OverWritesNotAllowedError,
    func_node_names_and_outs,
    get_first_item_and_assert_unicity,
)


def cached_dag_test():
    """
    Covering issue https://github.com/i2mint/meshed/issues/34
    about "CachedDag.cache should be populated with inputs that it was called on"
    """
    from meshed.dag import DAG

    def f(a, x=1):
        return a + x

    def g(a, y=2):
        return a * y

    dag = DAG([f, g])

    c = CachedDag(dag)
    assert c("g", a=1) == 2
    assert c.cache == {"g": 2, "a": 1}
    assert c("f") == 2


def add(a, b=1):
    return a + b


def mult(x, y=2):
    return x * y


def exp(mult, n=3):
    return mult**n


def subtract(a, b=4):
    return a - b
