"""Visualization utilities for the meshed package."""


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
    graph_widget = widgets.HTML(value=dot.pipe(format='svg').decode('utf-8'))
    display(graph_widget)

    def add_edge(sender):
        source = source_node.value
        target = target_node.value
        if (source, target) not in g.edges:
            g.add_edge(source, target)
            dot.edge(str(source), str(target))
            graph_widget.value = dot.pipe(format='svg').decode('utf-8')
        source_node.value = ''
        target_node.value = ''

    def add_node(sender):
        node = new_node.value
        if node not in g.nodes:
            g.add_node(node)
            dot.node(str(node))
            graph_widget.value = dot.pipe(format='svg').decode('utf-8')
        new_node.value = ''

    def delete_edge(sender):
        source = str(delete_source.value)
        target = str(delete_target.value)
        if (source, target) in g.edges:
            g.remove_edge(source, target)
            dot.body.remove(f'\t{source} -> {target}\n')
            graph_widget.value = dot.pipe(format='svg').decode('utf-8')
        delete_source.value = ''
        delete_target.value = ''

    def delete_node(sender):
        node = delete_node_value.value
        if node in g.nodes:
            g.remove_node(node)
            dot.body[:] = [line for line in dot.body if str(node) not in line]
            graph_widget.value = dot.pipe(format='svg').decode('utf-8')
        delete_node_value.value = ''

    source_node = widgets.Text(placeholder='Source Node')
    target_node = widgets.Text(placeholder='Target Node')
    add_edge_button = widgets.Button(description='Add Edge')
    add_edge_button.on_click(add_edge)

    new_node = widgets.Text(placeholder='New Node')
    add_node_button = widgets.Button(description='Add Node')
    add_node_button.on_click(add_node)

    delete_source = widgets.Text(placeholder='Source Node')
    delete_target = widgets.Text(placeholder='Target Node')
    delete_edge_button = widgets.Button(description='Delete Edge')
    delete_edge_button.on_click(delete_edge)

    delete_node_value = widgets.Text(placeholder='Node')
    delete_node_button = widgets.Button(description='Delete Node')
    delete_node_button.on_click(delete_node)

    controls = widgets.HBox([source_node, target_node, add_edge_button])
    controls2 = widgets.HBox([new_node, add_node_button])
    controls3 = widgets.HBox([delete_source, delete_target, delete_edge_button])
    controls4 = widgets.HBox([delete_node_value, delete_node_button])
    display(controls)
    display(controls2)
    display(controls3)
    display(controls4)
