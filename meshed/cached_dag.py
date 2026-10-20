"""Compute any var node of a DAG from what it depends on, caching what was computed.

:class:`CachedDag` wraps a :class:`~meshed.dag.DAG`: ``cached(k, **inputs)`` computes the
var node ``k`` from the given inputs, the cache, and the defaults, and keeps every input
and intermediate value for later calls. :meth:`CachedDag.inject_methods` gives an object
one method per computable var node, whose signature lists just the root inputs that
node still needs (i2mint/meshed#36).

>>> from meshed import DAG
>>> def f(a, x=1):
...     return a + x
>>> def g(a, y=2):
...     return a * y
>>> cached = CachedDag(DAG([f, g]))
>>> cached('g', a=1), cached('f')
(2, 2)
"""

from collections.abc import Mapping
from collections import ChainMap
from inspect import Signature
from meshed.itools import edge_reversed_graph, descendants
from i2 import Sig, Param, sort_params

__all__ = ["CachedDag", "NotAllowed", "OverWritesNotAllowedError", "NoOverwritesDict"]


class NotAllowed(Exception):
    """To use to indicate that something is not allowed"""


class OverWritesNotAllowedError(NotAllowed):
    """Error to raise when a writes to existing keys are not allowed"""


def get_first_item_and_assert_unicity(seq):
    seq_length = len(seq)
    if seq_length:
        assert seq_length == 1, (
            f"There should be one and one only item in the " f"sequence: {seq}"
        )
        return seq[0]
    else:
        return None


def func_node_names_and_outs(dag):
    for func_node in dag.func_nodes:
        yield func_node.name, func_node.out


class NoOverwritesDict(dict):
    """
    A dict where you're not allowed to write to a key that already has a value in it.

    >>> d = NoOverwritesDict(a=1, b=2)
    >>> d
    {'a': 1, 'b': 2}

    Writing is allowed, in new keys

    >>> d['c'] = 3
    >>> d
    {'a': 1, 'b': 2, 'c': 3}

    It's also okay to write into an existing key if the value it holds is identical.
    In fact, the write doesn't even happen.

    >>> d['b'] = 2

    But if we try to write a different value...

    >>> d['b'] = 22  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    cached_dag.OverWritesNotAllowedError: The b key already exists and you're not allowed to change its value
    """

    def __setitem__(self, key, value):
        if key not in self:
            super().__setitem__(key, value)
        elif value != self[key]:
            raise OverWritesNotAllowedError(
                f"The {key} key already exists and you're not allowed to change its "
                f"value"
            )
        # else, don't even write the value since it's the same


NoSuchKey = type("NoSuchKey", (), {})


def _is_same_value(a, b):
    """Whether ``a`` and ``b`` are the same value, never raising.

    Identity counts as sameness (so a ``nan`` object equals itself). Values whose
    ``==`` doesn't give a plain truth value (e.g. numpy arrays) are only considered
    the same if they are identical.

    >>> _is_same_value(1, 1), _is_same_value(1, 2)
    (True, False)
    >>> nan = float('nan')
    >>> _is_same_value(nan, nan), _is_same_value(nan, float('nan'))
    (True, False)
    """
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


# TODO: Cache validation and invalidation
# TODO: Continue constructing uppward towards lazyprop-using class (instances are
#  varnodes)
class CachedDag:
    """
    Wraps a DAG, using it to compute any of it's var nodes from it's dependents,
    with the capability of caching intermediate var nodes for later reuse.

    >>> def add(a, b=1):
    ...     return a + b
    >>> def mult(x, y=2):
    ...     return x * y
    >>> def subtract(a, b=4):
    ...     return a - b
    >>> from meshed import code_to_dag
    >>>
    >>> @code_to_dag(func_src=locals())
    ... def dag(w, ww, www):
    ...     x = mult(w, ww)
    ...     y = add(x, www)
    ...     z = subtract(x, y)
    >>> print(dag.dot_digraph_ascii())  # doctest: +SKIP

    .. code-block:: text

                        w

                     │
                     │
                     ▼
                   ┌──────────┐
         ww=   ──▶ │   mult   │
                   └──────────┘
                     │
                     │
                     ▼

                        x       ─┐
                                 │
                     │           │
                     │           │
                     ▼           │
                   ┌──────────┐  │
         www=  ──▶ │   add    │  │
                   └──────────┘  │
                     │           │
                     │           │
                     ▼           │
                                 │
                        y=       │
                                 │
                     │           │
                     │           │
                     ▼           │
                   ┌──────────┐  │
                   │ subtract │ ◀┘
                   └──────────┘
                     │
                     │
                     ▼

                        z

    >>> from inspect import signature
    >>> g = CachedDag(dag)
    >>> signature(g)
    <Signature (k, /, **input_kwargs)>

    We can get ``ww`` because it has a default: the dag's own signature
    (``w, ww, www``) has none, but ``ww`` feeds ``mult``'s ``y=2``, and a root takes
    the default of the parameters it binds (when they agree):

    >>> g.defaults
    {'ww': 2, 'www': 1}
    >>> g('ww')
    2

    But we can't get ``y`` because we don't have what it depends on:

    >>> g('y')
    Traceback (most recent call last):
        ...
    TypeError: The input_kwargs of a dag call is missing 1 required argument: 'w'

    It needs a ``w?``! No, it needs an ``x``! But to get an ``x`` you need a ``w``,
    and...

    >>> g('x')
    Traceback (most recent call last):
        ...
    TypeError: The input_kwargs of a dag call is missing 1 required argument: 'w'

    So let's give it a w!

    >>> g('x', w=3)  # == 3 * 2
    6

    And now this works:

    >>> g('x')
    6

    because

    >>> g.cache
    {'x': 6, 'w': 3}

    and this will work too:

    >>> g('y')
    7
    >>> g.cache
    {'x': 6, 'w': 3, 'y': 7}

    An input that contradicts the cache is refused rather than silently ignored:

    >>> g('x', w=10)
    Traceback (most recent call last):
        ...
    ValueError: input_kwargs can't contain keys that are already in cache with a different value! These names were in both: {'w'}

    Methods per var node, each asking only for the roots it still needs:

    >>> fresh = CachedDag(dag).inject_methods()
    >>> signature(fresh.y)
    <Signature (*, w, ww=2, www=1)>
    >>> fresh.x(w=3)
    6
    >>> signature(fresh.y)  # w is cached now
    <Signature (*, ww=2, www=1)>
    >>> fresh.z()
    -1
    """

    def __init__(self, dag, cache=True, name=None):
        self.dag = dag
        self.reversed_graph = edge_reversed_graph(dag.graph_ids)
        self.roots = set(self.dag.roots)
        self.leafs = set(self.dag.leafs)
        self.var_nodes = set(self.dag.var_nodes)
        self.func_node_of_id = {fn.out: fn for fn in self.dag.func_nodes}
        self.name = name
        self.out_of_func_node_name = dict(func_node_names_and_outs(self.dag))
        self._dag_sig = Sig(self.dag)
        self.defaults = {**_root_defaults(self.dag), **self._dag_sig.defaults}
        if cache is True:
            self.cache = NoOverwritesDict()
        elif not isinstance(cache, Mapping):
            raise NotImplementedError(
                "This type of cache is not implemented (must resolve to a Mapping): "
                f"{cache=}"
            )
        else:
            self.cache = cache
        # Cached values (computed outputs and inputs the dag was called with) take
        # precedence over the dag's defaults.
        self._cache = ChainMap(self.cache, self.defaults)

    @property
    def __name__(self):
        return self.name or self.dag.__name__

    def __iter__(self):
        yield from self.reversed_graph

    def func_node_id(self, k):
        func_node_name = get_first_item_and_assert_unicity(self.reversed_graph[k])
        if func_node_name is not None:
            return self.out_of_func_node_name[func_node_name]

    # TODO: Consider having args and kwargs instead of just input_kwargs.
    #   or making it (k, /, *args, **kwargs)
    def __call__(self, k, /, **input_kwargs):
        input_kwargs = dict(input_kwargs)
        self._validate_inputs_against_cache(input_kwargs)
        keys_before = set(self.cache)
        try:
            output = self._compute(k, input_kwargs)
        except BaseException:
            # Roll back the outputs computed during this failed call, so the cache
            # stays consistent with its (uncached) inputs and the call can be retried.
            try:
                for key in set(self.cache) - keys_before:
                    del self.cache[key]
            except Exception:  # pragma: no cover - e.g. a cache without __delitem__
                pass  # never mask the original error with a rollback error
            raise
        # Only persist inputs once they led to a successful computation, so that a
        # failed (e.g. mistaken) call doesn't pin values in the cache.
        self._cache_inputs(input_kwargs)
        return output

    def _validate_inputs_against_cache(self, input_kwargs):
        """Raise a ``ValueError`` if ``input_kwargs`` contradicts the cache.

        An input contradicts the cache if:

        - it is already cached with a different value, or
        - it is not cached, but cached outputs downstream of it were computed using
          its default (or it has no default), and the given value differs from it, or
        - it is not cached and not a root, but the cache (and the dag's defaults)
          already determine everything it would be computed from.

        Note that inputs are only validated against the *cache*: values given in the
        same call are not checked against each other (``c('h', f=100, a=1)`` is
        accepted even if ``f`` wouldn't be computed as ``100`` from ``a=1``).

        Values are compared with ``_is_same_value``: for values without a plain
        ``==`` truth value (e.g. numpy arrays), only the very same object counts as
        the same value.
        """
        conflicts = {
            name
            for name in input_kwargs.keys() & self.cache.keys()
            if not _is_same_value(input_kwargs[name], self.cache[name])
        }
        if conflicts:
            # TODO: Can give the user a more informative/correct message, since the
            #  user has more options than just the root nodes: They some combination of
            #  intermediates would also satisfy requirements.
            raise ValueError(
                f"input_kwargs can't contain keys that are already in cache with a "
                f"different value! These names were in both: {conflicts}"
            )
        for name in input_kwargs.keys() - self.cache.keys():
            if name not in self.var_nodes:
                continue
            cached_downstream = descendants(self.dag.graph_ids, [name]) & set(
                self.cache
            )
            if cached_downstream and not _is_same_value(
                input_kwargs[name], self.defaults.get(name, NoSuchKey)
            ):
                raise ValueError(
                    f"The value given for {name!r} contradicts the cache: "
                    f"{sorted(cached_downstream)} were already computed without it."
                )
            if name not in self.roots and self._is_determined_by_cache(name):
                cached_upstream = descendants(self.reversed_graph, [name]) & set(
                    self.cache
                )
                raise ValueError(
                    f"{name!r} is already determined by the cache "
                    f"({sorted(cached_upstream)}), so it can't be given as an input."
                )

    def _is_determined_by_cache(self, k):
        """Whether ``k``'s value is already fixed by the cache and the dag's defaults
        (that is, whether ``self(k)``, with no inputs, would give a value)."""
        if k in self.cache or k in self.defaults:
            return True
        func_node_id = self.func_node_id(k)
        if func_node_id is None:  # a root node with no value in sight
            return False
        func_node = self.func_node_of_id[func_node_id]
        return all(self._is_determined_by_cache(src) for src in func_node.bind.values())

    def _compute(self, k, input_kwargs):
        _cache = ChainMap(input_kwargs, self._cache)
        if k in _cache:
            return _cache[k]
        func_node_id = self.func_node_id(k)
        if func_node_id:
            if (output := self.cache.get(func_node_id)) is not None:
                return output
            else:
                func_node = self.func_node_of_id[func_node_id]
                input_sources = {
                    src: self._compute(src, input_kwargs)
                    for src in func_node.bind.values()
                }
                # TODO: do we need to include **self.defaults in the middle?
                inputs = ChainMap(_cache, input_sources)
                output = func_node.call_on_scope(inputs, write_output_into_scope=False)
                self.cache[func_node_id] = output
                return output
        else:  # k is a root node
            assert k in self.roots, f"Was expecting this to be a root node: {k}"
            inputs = ChainMap(input_kwargs, self._cache)
            if (output := inputs.get(k, NoSuchKey)) is not NoSuchKey:
                return output
            else:
                raise TypeError(
                    f"The input_kwargs of a {self.__name__} call is missing 1 required "
                    f"argument: '{k}'"
                )

    def _cache_inputs(self, input_kwargs):
        """Persist the (explicitly given) values of the dag's var nodes in the cache,
        so later calls can reuse them (see https://github.com/i2mint/meshed/issues/34).
        Keys that are not var nodes of the dag, or that are already cached (and were
        validated to hold the same value), are not written."""
        for name, value in input_kwargs.items():
            if name in self.var_nodes and name not in self.cache:
                self.cache[name] = value

    def _call(self, k, /, **kwargs):
        return self(k, **kwargs)

    def roots_for(self, node):
        """
        The set of roots that lead to ``node``.

        >>> from meshed.makers import code_to_dag
        >>> @code_to_dag
        ... def dag():
        ...     x = mult(w, ww)
        ...     y = add(x, www)
        ...     z = subtract(x, y)
        >>> print(dag.synopsis_string())
        w,ww -> mult -> x
        x,www -> add -> y
        x,y -> subtract -> z
        >>> g = CachedDag(dag)
        >>> sorted(g.roots_for('x'))
        ['w', 'ww']
        >>> sorted(g.roots_for('y'))
        ['w', 'ww', 'www']
        """
        return set(
            filter(self.roots.__contains__, descendants(self.reversed_graph, node))
        )

    def _signature_for_node_method(self, node):
        """The keyword-only roots ``node`` still needs (not cached), with their defaults."""
        order = {name: i for i, name in enumerate(self._dag_sig.names)}

        def gen():
            needed = sorted(
                (r for r in self.roots_for(node) if r not in self.cache),
                key=lambda name: (order.get(name, len(order)), name),
            )
            for name in needed:
                yield Param(
                    name=name,
                    kind=Param.KEYWORD_ONLY,
                    default=self.defaults.get(name, Param.empty),
                    annotation=self._dag_sig.annotations.get(name, Param.empty),
                )

        return Sig(sort_params(gen()))

    def inject_methods(self, obj=None):
        """Give ``obj`` (default: a new namespace) one method per non-root var node.

        ``obj.<node>(**roots)`` is ``self(<node>, **roots)``. Its signature is computed
        when asked, so it always lists the roots ``<node>`` still needs given the cache
        at that moment, with the defaults the dag (or the nodes it feeds) gives them.
        """
        non_root_var_nodes = sorted(
            filter(lambda x: x not in self.roots, self.var_nodes)
        )
        if obj is None:
            from types import SimpleNamespace

            obj = SimpleNamespace()
        for var_node in non_root_var_nodes:
            setattr(obj, var_node, _NodeMethod(self, var_node))
        obj._cache = self.cache
        return obj


class _NodeMethod:
    """``cached_dag(node, **kwargs)`` as a callable whose signature follows the cache."""

    def __init__(self, cached_dag: CachedDag, node: str):
        self._cached_dag, self._node = cached_dag, node
        self.__name__ = self.__qualname__ = node

    @property
    def __signature__(self) -> Signature:
        return Signature(self._cached_dag._signature_for_node_method(self._node).params)

    def __call__(self, **kwargs):
        return self._cached_dag(self._node, **kwargs)

    def __repr__(self):
        return f"<{type(self).__name__} {self._node}{self.__signature__}>"


def _root_defaults(dag) -> dict:
    """Defaults roots take from the function parameters they bind to, when those agree.

    A root bound to parameters with different defaults (or one without) gets none: no
    choice between them would be right.
    """
    roots = set(dag.roots)
    seen: dict = {}
    for func_node in dag.func_nodes:
        defaults = func_node.sig.defaults
        for param, var in func_node.bind.items():
            if var in roots:
                seen.setdefault(var, []).append(defaults.get(param, NoSuchKey))
    return {
        var: values[0]
        for var, values in seen.items()
        if values[0] is not NoSuchKey
        and all(_is_same_value(v, values[0]) for v in values[1:])
    }
