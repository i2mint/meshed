from typing import Mapping
from functools import partial
from collections import ChainMap
from meshed.itools import edge_reversed_graph, descendants
from i2 import Sig, Param, sort_params


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

    .. code-block::
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


    We can get ``ww`` because it has a default:

    (TODO: This (and further tests) stopped working since code_to_dag was enhanced
    with the ability to use the wrapped function's signature to determine the
    signature of the output dag. Need to fix this.)

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

    >>> g('x', w=3)  # == 3 * 2 ==
    6

    And now this works:

    >>> g('x')
    6

    because

    >>> g.cache
    {'x': 6}

    and this will work too:

    >>> g('y')
    7
    >>> g.cache
    {'x': 6, 'y': 7}

    But this is something we need to handle better!

    >>> g('x', w=10)
    6

    This is happending because there's already a x in the cache, and it takes precedence.
    This would be okay if consider CachedDag as a low level object that is never
    actually used by a user.
    But we need to protect the user from such effects!

    First, we probably should cache inputs too.

    The we can:
    - Make  computation take precedence over cache, overwriting the existing cache
        with the new resulting values

    - Allow the user to declare the entire cache, or just some variables in it,
    as write-once, to avoid creating bugs with the above proposal.

    - Cache multiple paths (lru_cache style) for different input combinations

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
        self.defaults = self._dag_sig.defaults
        if cache is True:
            self.cache = NoOverwritesDict()
        elif not isinstance(cache, Mapping):
            raise NotImplementedError(
                "This type of cache is not implemented (must resolve to a Mapping): "
                f"{cache=}"
            )
        self._cache = ChainMap(self.defaults, self.cache)

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
        #         print(f"Calling ({k=},{input_kwargs=})\t{self.cache=}")
        input_kwargs = dict(input_kwargs)
        if intersection := (input_kwargs.keys() & self.cache.keys()):
            # TODO: Can give the user a more informative/correct message, since the
            #  user has more options than just the root nodes: They some combination of
            #  intermediates would also satisfy requirements.
            raise ValueError(
                f"input_kwargs can't contain any keys that are already in cache! "
                f"These names were in both: {intersection}"
            )
        _cache = ChainMap(input_kwargs, self._cache)
        if k in _cache:
            return _cache[k]
        input_kwargs = dict(input_kwargs)
        func_node_id = self.func_node_id(k)
        #         print(f"{func_node_id=}")
        if func_node_id:
            if (output := self.cache.get(func_node_id)) is not None:
                return output
            else:
                func_node = self.func_node_of_id[func_node_id]
                input_sources = {
                    src: self(src, **input_kwargs) for src in func_node.bind.values()
                }
                #                 inputs = dict(input_sources, **input_kwargs)  #
                # TODO: do we need to include **self.defaults in the middle?
                inputs = ChainMap(_cache, input_sources)
                #                 print(f"Computing {func_node_id}: ", end=" ")
                output = func_node.call_on_scope(inputs, write_output_into_scope=False)
                self.cache[func_node_id] = output
                #                 print(f"result -> {output}")
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
        def gen():
            for name in filter(lambda x: x not in self.cache, self.roots_for(node)):
                yield Param(
                    name=name,
                    kind=Param.KEYWORD_ONLY,
                    default=self.defaults.get(name, Param.empty),
                    annotation=self._dag_sig.annotations.get(name, Param.empty),
                )

        return Sig(sort_params(gen()))

    def inject_methods(self, obj=None):
        # TODO: Should be input_names of reversed_graph, but resulting "shadow" in
        #  the root nodes, along with their defaults (filtered by cache)
        non_root_var_nodes = list(filter(lambda x: x not in self.roots, self.var_nodes))
        if obj is None:
            from types import SimpleNamespace

            obj = SimpleNamespace(**{k: None for k in non_root_var_nodes})
        for var_node in non_root_var_nodes:
            sig = self._signature_for_node_method(var_node)
            f = sig(partial(self._call, var_node))
            setattr(obj, var_node, f)

        obj._cache = self.cache
        return obj


def cached_dag_test():
    """
    Covering issue https://github.com/i2mint/meshed/issues/34
    about "CachedDag.cache should be populated with inputs that it was called on"
    """
    from meshed.dag import DAG

    def f(a, x=1):
        return a + x

    def g(a, y=2):
        return a * y

    dag = DAG([f, g])

    c = CachedDag(dag)
    c("g", a=1)
    assert c.cache == {"g": 2, "a": 1}
    assert c("f" == 2)


def add(a, b=1):
    return a + b


def mult(x, y=2):
    return x * y


def exp(mult, n=3):
    return mult**n


def subtract(a, b=4):
    return a - b


# from meshed import code_to_dag
#
#
# @code_to_dag(func_src=locals())
# def dag(w, ww, www):
#     x = mult(w, ww)
#     y = add(x, www)
#     z = subtract(x, y)
#
#
# g = CachedDag(dag)
#
# assert g('z', {'w': 2, 'ww': 3, 'www': 4}) == -4 == dag(2, 3, 4)
