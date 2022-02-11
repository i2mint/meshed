"""util functions"""
from typing import Iterable, Callable, Optional, Union

FunctionNamer = Callable[[Callable], str]


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


def name_of_obj(o) -> Union[str, None]:
    """
    Tries to find the (or "a") name for an object, even if `__name__` doesn't exist.

    >>> name_of_obj(map)
    'map'
    >>> name_of_obj([1, 2, 3])
    'list'
    >>> name_of_obj(print)
    'print'
    >>> name_of_obj(lambda x: x)
    '<lambda>'
    >>> from functools import partial
    >>> name_of_obj(partial(print, sep=','))
    'print'
    """
    if hasattr(o, '__name__'):
        return o.__name__
    elif hasattr(o, '__class__'):
        name = name_of_obj(o.__class__)
        if name == 'partial':
            if hasattr(o, 'func'):
                return name_of_obj(o.func)
        return name
    else:
        return None


def incremental_str_maker(str_format='{:03.f}'):
    """Make a function that will produce a (incrementally) new string at every call."""
    i = 0

    def mk_next_str():
        nonlocal i
        i += 1
        return str_format.format(i)

    return mk_next_str


lambda_name = incremental_str_maker(str_format='lambda_{:03.0f}')
unnameable_func_name = incremental_str_maker(str_format='unnameable_func_{:03.0f}')

func_name: FunctionNamer


def func_name(func) -> str:
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing"""
    try:
        name = func.__name__
        if name == '<lambda>':
            return lambda_name()
        return name
    except AttributeError:
        return unnameable_func_name()


# ---------------------------------------------------------------------------------------
# Misc

from typing import Iterable, Callable, Optional


def args_funcnames(
    funcs: Iterable[Callable], name_of_func: Optional[FunctionNamer] = func_name
):
    """Generates (arg_name, func_name) pairs from the iterable of functions"""
    from inspect import signature, Parameter

    for func in funcs:
        sig = signature(func)
        for param in sig.parameters.values():
            arg_name = ''  # initialize
            if param.kind == Parameter.VAR_POSITIONAL:
                arg_name += '*'
            elif param.kind == Parameter.VAR_KEYWORD:
                arg_name += '**'
            arg_name += param.name  # append name of param
            yield arg_name, name_of_func(func)


def funcs_to_digraph(funcs, graph=None):
    from graphviz import Digraph

    graph = graph or Digraph()
    graph.edges(list(args_funcnames(funcs)))
    graph.body.extend([', '.join(func.__name__ for func in funcs) + ' [shape=box]'])
    return graph


def print_ascii_graph(funcs):
    from lined.util import dot_to_ascii

    digraph = funcs_to_digraph(funcs)
    dot_str = '\n'.join(map(lambda x: x[1:], digraph.body[:-1]))
    print(dot_to_ascii(dot_str))
