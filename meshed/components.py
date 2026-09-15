"""Ready-made extraction components for meshed graphs.

A ``DAG`` node needs a ``__name__`` and a signature to know what var node it reads
and what var node it writes. Plain ``operator.itemgetter`` and ``attrgetter``
objects have neither, so this module wraps them in ``Extractor``, a callable that
carries a chosen name and a single positional-only parameter, and can therefore
be listed directly among the functions given to ``DAG``.

Main entry points:

- ``Itemgetter``: extracts one item, or a tuple of items, from its input by key.
- ``AttrGetter``: extracts one attribute, or a tuple of attributes, from its input.
- ``Extractor``: the general form; give it a factory and the parameters to build with.

>>> from meshed.components import Itemgetter
>>> get_ab = Itemgetter(['a', 'b'])
>>> get_ab({'a': 1, 'b': 2, 'c': 3})
(1, 2)
"""

from i2 import Sig
from typing import Any
from collections.abc import Callable
from operator import itemgetter, attrgetter
from dataclasses import dataclass
from functools import partial


@dataclass
class Extractor:
    """Callable extracting from its single input, named and signed to be a DAG node.

    Calling an instance applies ``extractor_factory(extractor_params)`` to the input.
    The instance carries a chosen ``__name__`` and a one-parameter signature, which
    is what ``DAG`` needs to wire it to var nodes.

    Args:
        extractor_factory: Called once with ``extractor_params`` to make the
            function that is applied to the input.
        extractor_params: Passed to ``extractor_factory``.
        name: Becomes the ``__name__`` of the instance.
        input_name: Name of the single positional-only parameter of the instance's
            signature (the var node a DAG will bind it to).
    """

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
