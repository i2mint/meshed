"""Visualization utilities for the meshed package."""

from typing import Iterable, Any
from i2.signatures import Sig


def dot_lines_of_objs(objs: Iterable, start_lines=(), end_lines=(), **kwargs):
    r"""
    Get lines generator for the graphviz.DiGraph(body=list(...))

    >>> from meshed.base import FuncNode
    >>> def add(a, b=1):
    ...     return a + b
    >>> def mult(x, y=3):
    ...     return x * y
    >>> def exp(mult, a):
    ...     return mult ** a
    >>> func_nodes = [
    ...     FuncNode(add, out='x'),
    ...     FuncNode(mult, name='the_product'),
    ...     FuncNode(exp)
    ... ]
    >>> lines = list(dot_lines_of_objs(func_nodes))
    >>> assert lines == [
    ... 'x [label="x" shape="none"]',
    ... '_add [label="_add" shape="box"]',
    ... '_add -> x',
    ... 'a [label="a" shape="none"]',
    ... 'b [label="b=" shape="none"]',
    ... 'a -> _add',
    ... 'b -> _add',
    ... 'mult [label="mult" shape="none"]',
    ... 'the_product [label="the_product" shape="box"]',
    ... 'the_product -> mult',
    ... 'x [label="x" shape="none"]',
    ... 'y [label="y=" shape="none"]',
    ... 'x -> the_product',
    ... 'y -> the_product',
    ... 'exp [label="exp" shape="none"]',
    ... '_exp [label="_exp" shape="box"]',
    ... '_exp -> exp',
    ... 'mult [label="mult" shape="none"]',
    ... 'a [label="a" shape="none"]',
    ... 'mult -> _exp',
    ... 'a -> _exp'
    ... ]  # doctest: +SKIP

    >>> from meshed.util import dot_to_ascii
    >>>
    >>> print(dot_to_ascii('\n'.join(lines)))  # doctest: +SKIP
    <BLANKLINE>
                    a        ─┐
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
     b=  ──▶ │    _add     │  │
             └─────────────┘  │
               │              │
               │              │
               ▼              │
                              │
                    x         │
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
     y=  ──▶ │ the_product │  │
             └─────────────┘  │
               │              │
               │              │
               ▼              │
                              │
                  mult        │
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
             │    _exp     │ ◀┘
             └─────────────┘
               │
               │
               ▼
    <BLANKLINE>
                   exp
    <BLANKLINE>

    """
    # Should we validate here, or outside this module?
    # from meshed.base import validate_that_func_node_names_are_sane
    # validate_that_func_node_names_are_sane(func_nodes)
    yield from start_lines
    for obj in objs:
        yield from obj.dot_lines(**kwargs)
    yield from end_lines


dot_lines_of_func_nodes = dot_lines_of_objs  # backwards compatiblity alias


# TODO: Should we integrate this to dot_lines_of_func_parameters directly (decorator?)
def add_new_line_if_none(s: str):
    """Since graphviz 0.18, need to have a newline in body lines.
    This util is there to address that, adding newlines to body lines
    when missing."""
    if s and s[-1] != "\n":
        return s + "\n"
    return s


# ------------------------------------------------------------------------------
# Unused -- consider deleting
def _parameters_and_names_from_sig(
    sig: Sig,
    out=None,
    func_name=None,
):
    func_name = func_name or sig.name
    out = out or sig.name
    if func_name == out:
        func_name = "_" + func_name
    assert isinstance(func_name, str) and isinstance(out, str)
    return sig.parameters, out, func_name


# ------------------------------------------------------------------------------
# Old stuff


def visualize_graph(graph):
    import graphviz
    from IPython.display import display

    dot = graphviz.Digraph()

    # Add nodes to the graph
    for node in graph:
        dot.node(node)

    # Add edges to the graph
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            dot.edge(node, neighbor)

    # Render and display the graph in the notebook
    display(dot)


def visualize_graph_interactive(graph):
    import graphviz

    import networkx as nx
    import ipywidgets as widgets
    from IPython.display import display

    g = nx.DiGraph(graph)

    # Create an empty Graphviz graph
    dot = graphviz.Digraph()

    # Add nodes to the Graphviz graph
    for node in g.nodes:
        dot.node(str(node))

    # Add edges to the Graphviz graph
    for edge in g.edges:
        dot.edge(str(edge[0]), str(edge[1]))

    # Render the initial graph visualization
    graph_widget = widgets.HTML(value=dot.pipe(format="svg").decode("utf-8"))
    display(graph_widget)

    def add_edge(sender):
        source = source_node.value
        target = target_node.value
        if (source, target) not in g.edges:
            g.add_edge(source, target)
            dot.edge(str(source), str(target))
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        source_node.value = ""
        target_node.value = ""

    def add_node(sender):
        node = new_node.value
        if node not in g.nodes:
            g.add_node(node)
            dot.node(str(node))
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        new_node.value = ""

    def delete_edge(sender):
        source = str(delete_source.value)
        target = str(delete_target.value)
        if (source, target) in g.edges:
            g.remove_edge(source, target)
            dot.body.remove(f"\t{source} -> {target}\n")
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        delete_source.value = ""
        delete_target.value = ""

    def delete_node(sender):
        node = delete_node_value.value
        if node in g.nodes:
            g.remove_node(node)
            dot.body[:] = [line for line in dot.body if str(node) not in line]
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        delete_node_value.value = ""

    source_node = widgets.Text(placeholder="Source Node")
    target_node = widgets.Text(placeholder="Target Node")
    add_edge_button = widgets.Button(description="Add Edge")
    add_edge_button.on_click(add_edge)

    new_node = widgets.Text(placeholder="New Node")
    add_node_button = widgets.Button(description="Add Node")
    add_node_button.on_click(add_node)

    delete_source = widgets.Text(placeholder="Source Node")
    delete_target = widgets.Text(placeholder="Target Node")
    delete_edge_button = widgets.Button(description="Delete Edge")
    delete_edge_button.on_click(delete_edge)

    delete_node_value = widgets.Text(placeholder="Node")
    delete_node_button = widgets.Button(description="Delete Node")
    delete_node_button.on_click(delete_node)

    controls = widgets.HBox([source_node, target_node, add_edge_button])
    controls2 = widgets.HBox([new_node, add_node_button])
    controls3 = widgets.HBox([delete_source, delete_target, delete_edge_button])
    controls4 = widgets.HBox([delete_node_value, delete_node_button])
    display(controls)
    display(controls2)
    display(controls3)
    display(controls4)
