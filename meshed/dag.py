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
from typing import Callable, Mapping, Optional, Iterable

from i2.signatures import Sig


def call_func_ignoring_excess(func, **kwargs):
    """Call func, sourcing the arguments from kwargs and ignoring the excess arguments.
    Also works if func has some position only arguments.
    """
    s = Sig(func)
    args, kwargs = s.args_and_kwargs_from_kwargs(s.source_kwargs(**kwargs))
    return func(*args, **kwargs)


@dataclass
class FuncNode:
    func: Callable
    src_names: dict = field(default_factory=dict)
    return_name: str = field(default=None)
    _return_name_suffix: str = field(default='__output', init=False, repr=False)

    def __post_init__(self):
        self.return_name = self.return_name or self.func.__name__ + self._return_name_suffix
        # wraps(self.func)(self)  # does what wraps does (a
        self.func_sig = Sig(self.func)
        self.func_sig(self)  # puts the signature of func on the call of self
        # self.src_names =

    def __call__(self, **src_kwargs):
        args, kwargs = self.func_sig.args_and_kwargs_from_kwargs(self.func_sig.source_kwargs(**src_kwargs))
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
            first_func.func_sig.source_kwargs(*args, **kwargs))
        d[first_func.return_name] = first_func(*args, **kwargs)
        for func in other_funcs:
            d[func.return_name] = func(**d)
        return d


from i2.signatures import Sig


def call_func(func, kwargs):
    kwargs = {k.__name__: v for k, v in kwargs.items()}
    return Sig(func).source_kwargs(kwargs)
