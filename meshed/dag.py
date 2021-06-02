from functools import partial, wraps


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


from dataclasses import dataclass, field
from typing import Callable, MutableMapping, Mapping, Optional, Iterable

from i2.signatures import call_forgivingly
from i2.deco import ch_func_to_all_pk


def hook_up(func, variables: MutableMapping, output_name=None):
    """Source inputs and write outputs to given variables mapping.

    Returns inputless and outputless function that will, when called,
    get relevant inputs from the provided variables mapping and write it's
    output there as well.

    :param variables: The MutableMapping (like... a dict) where the function
    should both read it's input and write it's output.
    :param output_name: The key of the variables mapping that should be used
    to write the output of the function
    :return: A function

    >>> def formula1(w, /, x: float, y=1, *, z: int = 1):
    ...     return ((w + x) * y) ** z

    >>> d = {}
    >>> f = hook_up(formula1, d)
    >>> # NOTE: update d, not d = dict(...), which would make a DIFFERENT d
    >>> d.update(w=2, x=3, y=4)  # not d = dict(w=2, x=3, y=4), which would
    >>> f()

    Note that there's no output. The output is in d
    >>> d
    {'w': 2, 'x': 3, 'y': 4, 'formula1': 20}

    Again...

    >>> d.clear()
    >>> d.update(w=1, x=2, y=3)
    >>> f()
    >>> d['formula1']
    9

    """
    _func = ch_func_to_all_pk(
        func
    )  # makes a position-keyword copy of func
    output_key = output_name
    if output_name is None:
        output_key = _func.__name__

    def source_from_decorated():
        variables[output_key] = call_forgivingly(_func, **variables)

    return source_from_decorated



# replaced by call_forgivingly
# def call_func_ignoring_excess(func, **kwargs):
#     """Call func, sourcing the arguments from kwargs and ignoring the excess arguments.
#     Also works if func has some position only arguments.
#     """
#     s = Sig(func)
#     args, kwargs = s.args_and_kwargs_from_kwargs(s.source_kwargs(**kwargs))
#     return func(*args, **kwargs)


@dataclass
class FuncNode:
    func: Callable
    src_names: dict = field(default_factory=dict)
    return_name: str = field(default=None)
    _return_name_suffix: str = field(
        default='__output', init=False, repr=False
    )

    def __post_init__(self):
        self.return_name = (
            self.return_name or self.func.__name__ + self._return_name_suffix
        )
        # wraps(self.func)(self)  # does what wraps does (a
        self.func_sig = Sig(self.func)
        self.func_sig(self)  # puts the signature of func on the call of self
        # self.src_names =

    def __call__(self, **src_kwargs):
        # return call_forgivingly(self. func, **src_kwargs)
        args, kwargs = self.func_sig.args_and_kwargs_from_kwargs(
            self.func_sig.source_kwargs(**src_kwargs)
        )
        return self.func(*args, **kwargs)


from meshed.makers import edge_reversed_graph


@dataclass
class DAG:
    func_nodes: Iterable[FuncNode]

    def __post_init__(self):
        first_func, *_ = self.func_nodes
        wraps(first_func)(self)

    def __call__(self, *args, **kwargs):
        d = dict()
        first_func, *other_funcs = self.func_nodes
        args, kwargs = first_func.func_sig.args_and_kwargs_from_kwargs(
            first_func.func_sig.source_kwargs(*args, **kwargs)
        )
        d[first_func.return_name] = first_func(*args, **kwargs)
        for func in other_funcs:
            d[func.return_name] = func(**d)
        return d


from i2.signatures import Sig


def call_func(func, kwargs):
    kwargs = {k.__name__: v for k, v in kwargs.items()}
    return Sig(func).source_kwargs(kwargs)
