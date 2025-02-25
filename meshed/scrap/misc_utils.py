"""Misc utils"""

from typing import Mapping
from meshed.util import ModuleNotFoundIgnore
from collections import deque, defaultdict

with ModuleNotFoundIgnore():
    import networkx as nx

    topological_sort_2 = nx.dag.topological_sort


from typing import Iterable


def mermaid_pack_nodes(
    mermaid_code: str,
    nodes: Iterable[str],
    packed_node_name: str = None,
    *,
    arrow: str = "-->",
) -> str:
    """
    Output mermaid code with nodes packed into a single node.

    >>> mermaid_code = '''
    ... graph TD
    ...   A --> B
    ...   B --> C
    ...   A --> D
    ...   D --> E
    ...   E --> C
    ... '''
    >>>
    >>>
    >>> print(mermaid_pack_nodes(mermaid_code, ['B', 'C', 'E'], 'BCE'))  # doctest: +NORMALIZE_WHITESPACE
    graph TD
    A -->BCE
    A --> D
    D -->BCE
    """
    if packed_node_name is None:
        packed_node_name = "__".join(nodes)

    def gen_lines():
        for line in mermaid_code.strip().split("\n"):
            source, _arrow, target = line.partition(arrow)
            if source.strip() in nodes:
                source = packed_node_name
            if target.strip() in nodes:
                target = packed_node_name
            if (source != target) and source != packed_node_name:
                yield f"{_arrow}".join([source, target])
            else:
                # If there are any loops within the nodes to be packed,
                # they'll be represented as a loop for the packed node, which we skip.
                continue

    return "\n".join(gen_lines())


from typing import Any, Mapping, Sized, MutableMapping, Iterable
from meshed.itools import children, parents


def coparents_sets(g: Mapping, source: Iterable):
    res = []
    for node in source:
        for kid in children(g, [node]):
            res.append(frozenset(parents(g, [kid])))
    return set(res)


def known_parents(g: Mapping, kid, source):
    return parents(g, [kid]).issubset(set(source))


def list_coparents(g: Mapping, coparent):
    all_kids = children(g, [coparent])
    result = [parents(g, [kid]) for kid in all_kids]

    return result


def kids_of_united_family(g: Mapping, source: Iterable):
    res = set()
    for coparent in source:
        for kid in children(g, [coparent]):
            if known_parents(g, kid, source):
                res.add(kid)
    return res


def extended_family(g: Mapping, source: Iterable):
    res = set(source)
    while True:
        allowed_kids = kids_of_united_family(g, res)
        if allowed_kids.issubset(res):
            return res
        res = res.union(allowed_kids)


from meshed.dag import DAG
from meshed.base import FuncNode


def funcnode_only(source: Iterable):
    return [item for item in source if isinstance(item, FuncNode)]


def dag_from_funcnodes(dag, input_names):
    kids = extended_family(g=dag.graph, source=input_names)
    fnodes = funcnode_only(kids)

    return DAG(fnodes)
