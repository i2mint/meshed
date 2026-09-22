"""Tests for ``meshed.scrap.cached_dag.CachedDag`` (see i2mint/meshed#34)."""

import pytest

from meshed import DAG
from meshed.scrap.cached_dag import CachedDag, cached_dag_test


def f(a, x=1):
    return a + x


def g(a, y=2):
    return a * y


def _cached_dag(**kwargs):
    return CachedDag(DAG([f, g]), **kwargs)


def test_cached_dag_test_function():
    cached_dag_test()


def test_inputs_are_cached_and_reused():
    c = _cached_dag()
    assert c("g", a=1) == 2
    assert c.cache == {"g": 2, "a": 1}
    assert c("f") == 2  # uses the cached ``a``
    assert c.cache == {"g": 2, "a": 1, "f": 2}


def test_repeating_an_input_with_the_same_value_is_allowed():
    c = _cached_dag()
    c("g", a=1)
    assert c("f", a=1) == 2


def test_conflicting_input_raises():
    c = _cached_dag()
    c("g", a=1)
    with pytest.raises(ValueError):
        c("f", a=10)


def test_cached_input_takes_precedence_over_default():
    c = _cached_dag()
    assert c("g", a=1, y=5) == 5
    assert c.cache["y"] == 5
    assert c("y") == 5  # not the dag's default (2)


def test_non_var_node_inputs_are_not_cached():
    c = _cached_dag()
    c("g", a=1, not_a_node=3)
    assert "not_a_node" not in c.cache


def test_failed_call_does_not_cache_inputs():
    c = _cached_dag()
    with pytest.raises(TypeError):
        c("g", y=5)  # missing ``a``
    assert c.cache == {}


def test_custom_mapping_cache():
    cache = {}
    c = _cached_dag(cache=cache)
    assert c("g", a=1) == 2
    assert cache == {"g": 2, "a": 1}


def k2(y, a):
    return y - a


def test_failed_call_does_not_cache_inputs_regardless_of_arg_order():
    c = CachedDag(DAG([k2]))
    with pytest.raises(TypeError):
        c("k2", y=5)  # missing ``a``, which comes *after* ``y``
    assert c.cache == {}


def h(f, g):
    return f + g


def test_multi_level_dag_with_values_without_plain_equality():
    class Arr:
        """Stand-in for a numpy array: ``==`` has an ambiguous truth value."""

        def __init__(self, v):
            self.v = v

        def __add__(self, other):
            return Arr(self.v + (other.v if isinstance(other, Arr) else other))

        __radd__ = __add__

        def __mul__(self, other):
            return Arr(self.v * other)

        def __eq__(self, other):
            class Ambiguous:
                def __bool__(self):
                    raise ValueError("ambiguous")

            return Ambiguous()

        __ne__ = __eq__

    c = CachedDag(DAG([f, g, h]))
    a = Arr(1)
    assert c("h", a=a).v == 4  # (1 + 1) + (1 * 2)
    assert c.cache["a"] is a
    assert c("h", a=a).v == 4  # same object again: allowed


def test_nan_input():
    nan = float("nan")
    c = _cached_dag()
    out = c("f", a=nan)
    assert out != out  # nan
    c("g", a=nan)  # the same nan object is accepted again


def test_input_contradicting_cached_outputs_raises():
    c = _cached_dag()
    c("f", a=1)  # computed with the default x=1
    assert c("f", a=1, x=1) == 2  # same as the default used: fine
    with pytest.raises(ValueError):
        c("f", a=1, x=5)  # f was computed with x=1


def test_intermediate_node_as_input():
    c = CachedDag(DAG([f, g, h]))
    assert c("h", f=100, a=1) == 102
    assert c.cache["f"] == 100
    with pytest.raises(ValueError):
        c("h", f=5)


def k(f, b):
    return f + b


def test_failed_call_can_be_retried():
    c = CachedDag(DAG([f, k]))
    with pytest.raises(TypeError):
        c("k", a=1)  # missing ``b`` (after ``f`` was computed)
    assert c.cache == {}  # partial results were rolled back
    assert c("k", a=1, b=3) == 5
    assert c.cache == {"f": 2, "k": 5, "a": 1, "b": 3}


def test_exception_in_function_can_be_retried():
    calls = []

    def flaky(a):
        calls.append(a)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return a

    def total(f, flaky):
        return f + flaky

    c = CachedDag(DAG([f, flaky, total]))
    with pytest.raises(RuntimeError):
        c("total", a=1)
    assert c("total", a=1) == 3


def test_input_determined_by_cached_upstream_values_raises():
    c = CachedDag(DAG([f, g, h]))
    c("f", a=1)  # caches a=1, which determines g (and therefore h)
    with pytest.raises(ValueError):
        c("h", g=99)
