"""Misc utils"""

from typing import Mapping
from meshed.util import ModuleNotFoundIgnore
from collections import deque, defaultdict

with ModuleNotFoundIgnore():
    import networkx as nx

    topological_sort_2 = nx.dag.topological_sort


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
