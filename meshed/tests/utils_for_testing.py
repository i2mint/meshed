"""Make objects for testing fast"""

from meshed.dag import *
from i2 import Sig


def parse_names(string):
    return list(map(str.strip, string.split(',')))


def mk_func(arg_names, func_name):
    sig = Sig(arg_names)

    @sig
    def func(*args, **kwargs):
        _kwargs = sig.kwargs_from_args_and_kwargs(args, kwargs)
        _kwargs_str = ', '.join(f'{k}={v}' for k, v in _kwargs.items())
        return f'{func_name}({_kwargs_str})'

    func.__name__ = func_name

    return func


def string_to_func(dot_string):
    arg_names, func_name = map(parse_names, dot_string.split('->'))
    assert len(func_name) == 1
    func_name = func_name[0]
    return mk_func(arg_names, func_name)


def string_to_func_node(dot_string):
    arg_names, func_name, output_name = map(parse_names, dot_string.split('->'))
    assert len(func_name) == 1
    func_name = func_name[0]
    assert len(output_name) == 1
    output_name = output_name[0]

    func = mk_func(arg_names, func_name)
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
    b,f -> g -> h
    a,b,c -> d -> e
    a,e -> i -> j
    >>> Sig(dag)
    <Sig (b, f, a, c)>
    >>> sorted(dag(1,2,3,4))
    ['g(b=1, f=2)', 'i(a=3, e=d(a=3, b=1, c=4))']
    """
    func_nodes = list(map(string_to_func_node, filter(bool, dot_string.split('\n'))))
    return DAG(func_nodes)
