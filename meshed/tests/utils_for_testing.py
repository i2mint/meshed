"""Make objects for testing fast"""

from meshed.util import mk_place_holder_func
from meshed.dag import *
from i2 import Sig


def parse_names(string):
    return list(map(str.strip, string.split(",")))


def string_to_func(dot_string):
    arg_names, func_name = map(parse_names, dot_string.split("->"))
    assert len(func_name) == 1
    func_name = func_name[0]
    return mk_place_holder_func(arg_names, func_name)


def string_to_func_node(dot_string):
    arg_names, func_name, output_name = map(parse_names, dot_string.split("->"))
    assert len(func_name) == 1

    func_name = func_name[0]
    assert len(output_name) == 1
    output_name = output_name[0]

    func = mk_place_holder_func(arg_names, func_name)
    return FuncNode(func, name=func_name, out=output_name)


def string_to_dag(dot_string):
    """
    >>> dot_string = '''
    ... a, b, c -> d -> e
    ... b, f -> g -> h
    ... a, e -> i -> j
    ... '''
    >>> dag = string_to_dag(dot_string)
    >>> print(dag.synopsis_string())
    a,b,c -> d -> e
    b,f -> g -> h
    a,e -> i -> j

    >>> Sig(dag)
    <Sig (a, b, c, f)>
    >>> sorted(dag(1,2,3,4))
    ['g(b=2, f=4)', 'i(a=1, e=d(a=1, b=2, c=3))']"""
    func_nodes = list(map(string_to_func_node, filter(bool, dot_string.split("\n"))))
    return DAG(func_nodes)
