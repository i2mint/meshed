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
