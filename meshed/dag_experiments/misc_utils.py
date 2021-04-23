from meshed.util import ModuleNotFoundIgnore
from collections import deque


def topological_sort_1(self, graph=None):
    """ Returns a topological ordering of the DAG.
    Raises an error if this is not possible (graph is not valid).
    """
    if graph is None:
        graph = self.graph

    in_degree = {}
    for u in graph:
        in_degree[u] = 0

    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque()
    for u in in_degree:
        if in_degree[u] == 0:
            queue.appendleft(u)

    l = []
    while queue:
        u = queue.pop()
        l.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.appendleft(v)

    if len(l) == len(graph):
        return l
    else:
        raise ValueError('graph is not acyclic')


with ModuleNotFoundIgnore():
    import networkx as nx

    topological_sort_2 = nx.dag.topological_sort
