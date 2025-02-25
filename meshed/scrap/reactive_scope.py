"""
Ideas towards a reactive-programming interpretation of meshes.
A scope (MutableMapping --think dict-like) that reacts to writes by computing
associated functions, themselves writing in the scope, creating a chain reaction that
propagates information through the scope.
"""

from functools import cached_property

from meshed import DAG, FuncNode

# TODO: Should ReactiveFuncNode exist? Could put the logic in a function and used in
#  ReactiveScope instead.


class ReactiveFuncNode(FuncNode):
    """A ``FuncNode`` that computes on a scope only if the scope has what it takes"""

    @cached_property
    def _dependencies(self):
        """The keys the scope needs to have so that the FuncNode is callable"""
        return set(self.bind.values())

    def call_on_scope(self, scope, write_output_into_scope=True):
        if self._dependencies.issubset(scope):
            return super().call_on_scope(scope, write_output_into_scope)


from typing import MutableMapping


# TODO: Don't seem to need the relations to be acyclic. Try it out, and make it work.
# TODO: If we allow multiple writes/deletes, to a key or even to different keys,
#  we open ourselves to a lot more complexity. We need to be able to detect when
#  existing values are not valid anymore, given the relations exhibited by the func
#  nodes. This is a lot of work, and I'm not sure it's worth it. Might be better to
#  keep the scope as a simple mapping, and protect it from having actions taken on it
#  that might bring it into an invalid state.
class ReactiveScope(MutableMapping):
    """
    A scope that reacts to writes by computing associated functions, themselves writing
    in the scope, creating a chain reaction that propagates information through the
    scope.

    Parameters
    ----------
    func_nodes : Iterable[ReactiveFuncNode]
        The functions that will be called when the scope is written to.
    scope_factory : Callable[[], MutableMapping]
        A factory that returns a new scope. The scope will be cleared by calling this
        factory at each call to `.clear()`.

    Examples
    --------

    First, we need some func nodes to define the reaction relationships.
    We'll stuff these func nodes in a DAG, for ease of use, but it's not necessary.

    >>> from meshed import FuncNode, DAG
    >>>
    >>> def f(a, b):
    ...     return a + b
    >>> def g(a_plus_b, d):
    ...     return a_plus_b * d
    >>> f_node = FuncNode(func=f, out='a_plus_b')
    >>> g_node = FuncNode(func=g, bind={'d': 'b'})
    >>> d = DAG((f_node, g_node))
    >>>
    >>> print(d.dot_digraph_ascii())
    <BLANKLINE>
                  a
    <BLANKLINE>
                │
                │
                ▼
              ┌────────┐
      b   ──▶ │   f    │
              └────────┘
      │         │
      │         │
      │         ▼
      │
      │        a_plus_b
      │
      │         │
      │         │
      │         ▼
      │       ┌────────┐
      └─────▶ │   g_   │
              └────────┘
                │
                │
                ▼
    <BLANKLINE>
                  g
    <BLANKLINE>

    Now we make a scope with these func nodes.

    >>> s = ReactiveScope(d)

    The scope starts empty (by default).

    >>> s
    <ReactiveScope with .scope: {}>

    So if we try to access any key, we'll get a KeyError.

    >>> s['g']  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    KeyError: 'g'

    That's because we didn't put write anything in the scope yet.

    But, if you give ``g_`` enough data to be able to compute ``g`` (namely, if you
    write values of ``b`` and ``a_plus_b``), then ``g`` will automatically be computed.

    >>> s['b'] = 3
    >>> s['a_plus_b'] = 5
    >>> s
    <ReactiveScope with .scope: {'b': 3, 'a_plus_b': 5, 'g': 15}>

    So now we can access ``g``.

    >>> s['g']
    15

    Note though, that we first showed that ``g`` appeared in the scope before we
    explicitly asked for it. This was to show that ``g`` was computed as a
    side-effect of writing to the scope, not because we asked for it, triggering the
    computation

    Let's clear the scope and show that by specifying ``a`` and ``b``, we get all the
    other values of the network.

    >>> s.clear()
    >>> s
    <ReactiveScope with .scope: {}>
    >>> s['a'] = 3
    >>> s['b'] = 4
    >>> s
    <ReactiveScope with .scope: {'a': 3, 'b': 4, 'a_plus_b': 7, 'g': 28}>
    >>> s['g']  # (3 + 4) * 4 == 7 * 4 == 28
    28
    """

    def __init__(self, func_nodes=(), scope_factory=dict):
        # Note: scope_factory could be made to return a pre-filled dict too
        if isinstance(func_nodes, DAG):
            dag = func_nodes
            func_nodes = dag.func_nodes
        func_nodes = [ReactiveFuncNode.from_dict(fn.to_dict()) for fn in func_nodes]
        self.dag = DAG(func_nodes)
        self.func_nodes_for_var_node = {
            k: v for k, v in self.dag.graph.items() if k in self.dag.var_nodes
        }
        self.scope_factory = scope_factory
        self.clear()

    def clear(self):
        """Note: This actually doesn't clear the mapping, but rather, resets it to it's original state,
        as defined by the `.scope_factory`"""
        self.scope = self.scope_factory()

    def __getitem__(self, k):
        # TODO: try/catch and give the user a bit more info (e.g. what dependencies are missing?)
        return self.scope[k]

    def __setitem__(self, k, v):
        # write the value under the key
        self.scope[k] = v
        # TODO: Need to make sure the func_node are in topological order
        # TODO: The .get(k, ()): prefill with missing keys at init time instead?
        for func_node in self.func_nodes_for_var_node.get(k, ()):
            # "try" calling the func_node on the scope (if scope doesn't have enough
            #
            func_node.call_on_scope(self.scope)

    def __len__(self):
        return len(self.scope)

    def __contains__(self, k):
        return k in self.scope

    def __iter__(self):
        return iter(self.scope)

    def __delitem__(self, k):
        # TODO: Could use the same mechanism as setitem to propagate the deletion through the network
        raise NotImplementedError(
            "deletion of keys are not implemented, since cache invalidation hasn't. "
            "You can clear the whole scope with the `.clear()` method. "
            "(Note: This actually doesn't clear the mapping, but rather, resets it to it's original state.)"
        )

    def __repr__(self):
        return f"<{type(self).__qualname__} with .scope: {repr(self.scope)}>"
