"""util functions"""
from functools import partial
from typing import Iterable, Callable, Optional, Union

from i2 import Sig

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


class ValidationError(ValueError):
    """Error that is raised when an object's validation failed"""


class NotUniqueError(ValidationError):
    """Error to be raised when unicity is expected, but violated"""


class NotFound(ValidationError):
    """To be raised when something is expected to exist, but doesn't"""


class NameValidationError(ValueError):
    """Use to indicate that there's a problem with a name or generating a valid name"""


def find_first_free_name(prefix, exclude_names=(), start_at=2):
    if prefix not in exclude_names:
        return prefix
    else:
        i = start_at
        while True:
            name = f'{prefix}__{i}'
            if name not in exclude_names:
                return name
            i += 1


def mk_func_name(func, exclude_names=()):
    name = getattr(func, '__name__', '')
    if name == '<lambda>':
        name = lambda_name()  # make a lambda name that is a unique identifier
    elif name == '':
        if isinstance(func, partial):
            return mk_func_name(func.func, exclude_names)
        else:
            raise NameValidationError(f"Can't make a name for func: {func}")
    return find_first_free_name(name, exclude_names)


def arg_names(func, func_name, exclude_names=()):
    names = Sig(func).names

    def gen():
        _exclude_names = exclude_names
        for name in names:
            if name not in _exclude_names:
                yield name
            else:
                found_name = find_first_free_name(
                    f'{func_name}__{name}', _exclude_names
                )
                yield found_name
                _exclude_names = _exclude_names + (found_name,)

    return list(gen())


def named_partial(func, *args, __name__=None, **keywords):
    """functools.partial, but with a __name__

    >>> f = named_partial(print, sep='\\n')
    >>> f.__name__
    'print'

    >>> f = named_partial(print, sep='\\n', __name__='now_partial_has_a_name')
    >>> f.__name__
    'now_partial_has_a_name'
    """
    f = partial(func, *args, **keywords)
    f.__name__ = __name__ or func.__name__
    return f
