"""Misc utils"""

from typing import Mapping
from meshed.util import ModuleNotFoundIgnore
from collections import deque, defaultdict

with ModuleNotFoundIgnore():
    import networkx as nx

    topological_sort_2 = nx.dag.topological_sort
