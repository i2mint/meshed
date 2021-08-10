import meshed as ms
import pytest


@pytest.fixture
def simple_graph():
    return dict(a="c", b="cd", c="abd", e="")


def test_edge_reversed_graph(simple_graph):
    g = simple_graph
    assert ms.makers.edge_reversed_graph(g) == {
        "c": ["a", "b"],
        "d": ["b", "c"],
        "a": ["c"],
        "b": ["c"],
        "e": [],
    }
    reverse_g_with_sets = ms.makers.edge_reversed_graph(g, set, set.add)
    assert reverse_g_with_sets == {
        "c": {"a", "b"},
        "d": {"b", "c"},
        "a": {"c"},
        "b": {"c"},
        "e": set([]),
    }
    assert ms.makers.edge_reversed_graph(dict(e="", a="e")) == {
        "e": ["a"],
        "a": [],
    }
    assert ms.makers.edge_reversed_graph(dict(a="e", e="")) == {
        "e": ["a"],
        "a": [],
    }
