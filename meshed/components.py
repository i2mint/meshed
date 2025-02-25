"""
Specialized components for meshed.
"""

from i2 import Sig
from typing import Callable, Any
from operator import itemgetter, attrgetter
from dataclasses import dataclass
from functools import partial


@dataclass
class Extractor:
    extractor_factory: Callable[[Any], Callable]
    extractor_params: Any
    # TODO: When migrating CI to 3.10+, can use `kw_only=True` here
    # name: str = field(default='extractor', kw_only=True)
    # input_name: str = field(default='x', kw_only=True)
    # But meanwhile, need an actual __init__ method:

    def __init__(
        self,
        extractor_factory: Callable[[Any], Callable],
        extractor_params: Any,
        *,
        name: str = "extractor",
        input_name: str = "x",
    ):
        self.extractor_factory = extractor_factory
        self.extractor_params = extractor_params
        self.name = name
        self.input_name = input_name
        self.__post_init__()

    def __post_init__(self):
        self.__name__ = self.name
        self.__signature__ = Sig(f"({self.input_name}, /)")
        self._call = self.extractor_factory(self.extractor_params)

    def __call__(self, x):
        return self._call(x)


def _itemgetter(items):
    if isinstance(items, str):
        items = [items]
    return itemgetter(*items)


def _attrgetter(attrs):
    if isinstance(attrs, str):
        attrs = [attrs]
    return attrgetter(*attrs)


Itemgetter = partial(Extractor, _itemgetter, name="itemgetter")
AttrGetter = partial(Extractor, _attrgetter, name="attrgetter")
