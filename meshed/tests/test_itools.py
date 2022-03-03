import meshed as ms
import pytest


@pytest.fixture
def example_graph():
    return dict(a=['c'], b=['c', 'c'], c=['a', 'b', 'd', 'e'], d=['c'], e=['c', 'z'])


@pytest.fixture
def graph_children():
    return {0: [1, 2], 1: [2, 3, 4], 2: [1, 4], 3: [4]}


@pytest.fixture
def digraph_children():
    return {0: [1, 2], 1: [2, 3, 4], 2: [4], 3: [4]}


@pytest.fixture
def graph_dict():
    return dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})


def test_add_edge(example_graph):
    g = example_graph
    ms.itools.add_edge(g, 'd', 'a')
    assert g == {
        'a': ['c'],
        'b': ['c', 'c'],
        'c': ['a', 'b', 'd', 'e'],
        'd': ['c', 'a'],
        'e': ['c', 'z'],
    }
    ms.itools.add_edge(g, 't', 'y')
    assert g == {
        'a': ['c'],
        'b': ['c', 'c'],
        'c': ['a', 'b', 'd', 'e'],
        'd': ['c', 'a'],
        'e': ['c', 'z'],
        't': ['y'],
    }


def test_edges(example_graph):
    g = example_graph
    assert sorted(ms.itools.edges(g)) == [
        ('a', 'c'),
        ('b', 'c'),
        ('b', 'c'),
        ('c', 'a'),
        ('c', 'b'),
        ('c', 'd'),
        ('c', 'e'),
        ('d', 'c'),
        ('e', 'c'),
        ('e', 'z'),
    ]


def test_nodes(example_graph):
    g = example_graph
    assert sorted(ms.itools.nodes(g)) == ['a', 'b', 'c', 'd', 'e', 'z']


def test_has_node():
    g = {0: [1, 2], 1: [2]}
    assert ms.itools.has_node(g, 0)
    assert ms.itools.has_node(g, 2)
    assert not ms.itools.has_node(g, 2, check_adjacencies=False)
    gg = {0: [1, 2], 1: [2], 2: []}
    assert ms.itools.has_node(gg, 2, check_adjacencies=False)


def test_successors(graph_children):
    g = graph_children
    assert set(ms.itools.successors(g, 1)) == {1, 2, 3, 4}
    assert set(ms.itools.successors(g, 3)) == {4}
    assert set(ms.itools.successors(g, 4)) == set()


def test_predecessors(graph_children):
    g = graph_children
    assert set(ms.itools.predecessors(g, 4)) == {0, 1, 2, 3}
    assert set(ms.itools.predecessors(g, 2)) == {0, 1, 2}
    assert set(ms.itools.predecessors(g, 0)) == set()


def test_children(graph_children):
    g = graph_children
    assert set(ms.itools.children(g, [2, 3])) == {1, 4}
    assert set(ms.itools.children(g, [4])) == set()


def test_parents(graph_children):
    g = graph_children
    assert set(ms.itools.parents(g, [2, 3])) == {0, 1}
    assert set(ms.itools.parents(g, [0])) == set()


def test_ancestors(digraph_children):
    g = digraph_children
    assert set(ms.itools.ancestors(g, [2, 3])) == {0, 1}
    assert set(ms.itools.ancestors(g, [0])) == set()


def test_descendants(digraph_children):
    g = digraph_children
    assert set(ms.itools.descendants(g, [2, 3])) == {4}
    assert set(ms.itools.descendants(g, [4])) == set()


def test_root_nodes(graph_dict):
    g = graph_dict
    assert sorted(ms.itools.root_nodes(g)) == ['f']


def test_leaf_nodes(graph_dict):
    g = graph_dict
    assert sorted(ms.itools.leaf_nodes(g)) == ['f', 'z']


def test_isolated_nodes(graph_dict):
    g = graph_dict
    assert set(ms.itools.isolated_nodes(g)) == {'f'}


def test_find_path(graph_dict):
    g = graph_dict
    assert ms.itools.find_path(g, 'a', 'c') == ['a', 'c']
    assert ms.itools.find_path(g, 'a', 'b') == ['a', 'c', 'b']
    assert ms.itools.find_path(g, 'a', 'z') == ['a', 'c', 'b', 'e', 'z']


def test_reverse_edges(example_graph):
    g = example_graph
    assert sorted(list(ms.itools.reverse_edges(g))) == [
        ('a', 'c'),
        ('b', 'c'),
        ('c', 'a'),
        ('c', 'b'),
        ('c', 'b'),
        ('c', 'd'),
        ('c', 'e'),
        ('d', 'c'),
        ('e', 'c'),
        ('z', 'e'),
    ]


def test_out_degrees(graph_dict):
    g = graph_dict
    assert dict(ms.itools.out_degrees(g)) == (
        {'a': 1, 'b': 2, 'c': 4, 'd': 1, 'e': 2, 'f': 0}
    )


def test_in_degrees(graph_dict):
    g = graph_dict
    assert dict(ms.itools.in_degrees(g)) == (
        {'a': 1, 'b': 1, 'c': 4, 'd': 1, 'e': 2, 'f': 0, 'z': 1}
    )


def test_copy_of_g_with_some_keys_removed(example_graph):
    g = example_graph
    keys = ['c', 'd']
    gg = ms.itools.copy_of_g_with_some_keys_removed(g, keys)
    assert gg == {'a': ['c'], 'b': ['c', 'c'], 'e': ['c', 'z']}


def test_topological_sort_helper(example_graph):
    g = example_graph
    v = 'a'
    stack = ['b', 'c']
    visited = set(['e'])
    ms.itools._topological_sort_helper(g, v, visited, stack)
    assert visited == {'a', 'b', 'c', 'd', 'e'}
    assert sorted(stack) == ['a', 'b', 'b', 'c', 'c', 'd']


def test_topological_sort():
    g = {0: [4, 2], 4: [3, 1], 2: [3], 3: [1]}
    assert list(ms.itools.topological_sort(g)) == [0, 2, 4, 3, 1]


def test_handle_exclude_nodes():
    def f(a=1, b=2, _exclude_nodes=['c']):
        return f'_exclude_nodes is now a set:{_exclude_nodes}'

    new_f = ms.itools._handle_exclude_nodes(f)
    assert new_f(1) == "_exclude_nodes is now a set:{'c'}"


@pytest.fixture
def simple_graph():
    return dict(a='c', b='cd', c='abd', e='')


def test_edge_reversed_graph(simple_graph):
    g = simple_graph
    assert ms.itools.edge_reversed_graph(g) == {
        'c': ['a', 'b'],
        'd': ['b', 'c'],
        'a': ['c'],
        'b': ['c'],
        'e': [],
    }
    reverse_g_with_sets = ms.itools.edge_reversed_graph(g, set, set.add)
    assert reverse_g_with_sets == {
        'c': {'a', 'b'},
        'd': {'b', 'c'},
        'a': {'c'},
        'b': {'c'},
        'e': set([]),
    }
    assert ms.itools.edge_reversed_graph(dict(e='', a='e')) == {
        'e': ['a'],
        'a': [],
    }
    assert ms.itools.edge_reversed_graph(dict(a='e', e='')) == {
        'e': ['a'],
        'a': [],
    }
