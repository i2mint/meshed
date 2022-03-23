"""
Making DAGs

In it's simplest form, consider this:

>>> from meshed import DAG
>>>
>>> def this(a, b=1):
...     return a + b
...
>>> def that(x, b=1):
...     return x * b
...
>>> def combine(this, that):
...     return (this, that)
...
>>>
>>> dag = DAG((this, that, combine))
>>> print(dag.synopsis_string())
x,b -> that_ -> that
a,b -> this_ -> this
this,that -> combine_ -> combine

But don't be fooled: There's much more to it!


FAQ and Troubleshooting
=======================

DAGs and Pipelines
------------------

>>> from functools import partial
>>> from meshed import DAG
>>> def chunker(sequence, chk_size: int):
...     return zip(*[iter(sequence)] * chk_size)
>>>
>>> my_chunker = partial(chunker, chk_size=3)
>>>
>>> vec = range(8)  # when appropriate, use easier to read sequences
>>> list(my_chunker(vec))
[(0, 1, 2), (3, 4, 5)]

Oh, that's just a ``my_chunker -> list`` pipeline!
A pipeline is a subset of DAG, so let me do this:

>>> dag = DAG([my_chunker, list])
>>> dag(vec)
Traceback (most recent call last):
...
TypeError: missing a required argument: 'sequence'

What happened here?
You're assuming that saying ``[my_chunker, list]`` is enough for DAG to know that
what you meant is for ``my_chunker`` to feed it's input to ``list``.
Sure, DAG has enough information to do so, but the default connection policy doesn't
assume that it's a pipeline you want to make.
In fact, the order you specify the functions doesn't have an affect on the connections
with the default connection policy.

See what the signature of ``dag`` is:

>>> from inspect import signature
>>> str(signature(dag))
'(iterable=(), /, sequence, *, chk_size: int = 3)'

So dag actually works just fine. Here's the proof:

>>> dag([1,2,3], vec)  # doctest: +SKIP
([1, 2, 3], <zip object at 0x104d7f080>)

It's just not what you might have intended.

Your best bet to get what you intended is to be explicit.

The way to be explicit is to not specify functions alone, but ``FuncNodes`` that
wrap them, along with the specification
the ``name`` the function will be referred to by,
the names that it's parameters should ``bind`` to (that is, where the function
will get it's import arguments from), and
the ``out`` name of where it should be it's output.

In the current case a fully specified DAG would look something like this:

>>> from meshed import FuncNode
>>> dag = DAG(
...     [
...         FuncNode(
...             func=my_chunker,
...             name='chunker',
...             bind=dict(sequence='sequence', chk_size='chk_size'),
...             out='chks'
...         ),
...         FuncNode(
...             func=list,
...             name='gather_chks_into_list',
...             bind=dict(iterable='chks'),
...             out='list_of_chks'
...         ),
...     ]
... )
>>> list(dag(vec))
[(0, 1, 2), (3, 4, 5)]

But really, if you didn't care about the names of things,
all you need in this case was to make sure that the output of ``my_chunker`` was
fed to ``list``, and therefore the following was sufficient:

>>> dag = DAG([
...     FuncNode(my_chunker, out='chks'),  # call the output of chunker "chks"
...     FuncNode(list, bind=dict(iterable='chks'))  # source list input from "chks"
... ])
>>> list(dag(vec))
[(0, 1, 2), (3, 4, 5)]

Connection policies are very useful when you want to define ways for DAG to
"just figure it out" for you.
That is, you want to tell the machine to adapt to your thoughts, not vice versa.
We support such technological expectations!
The default connection policy is there to provide one such ways, but
by all means, use another!

Does this mean that connection policies are not for production code?
Well, it depends. The Zen of Python (``import this``)
states "explicit is better than implicit", and indeed it's often
a good fallback rule.
But defining components and the way they should be assembled can go a long way
in achieving consistency, separation of concerns, adaptability, and flexibility.
All quite useful things. Also in production. Especially in production.
That said it is your responsiblity to use the right policy for your particular context.

"""

from contextlib import suppress
from functools import partial, wraps
from collections import Counter, defaultdict

from dataclasses import dataclass, field
from typing import Callable, MutableMapping, Sized, Union, Optional, Iterable, Any

from i2.signatures import (
    call_somewhat_forgivingly,
    Parameter,
    empty,
    Sig,
    sort_params,
)
from meshed.base import (
    FuncNode,
    validate_that_func_node_names_are_sane,
    _mk_func_nodes,
    _func_nodes_to_graph_dict,
    is_func_node,
)

from meshed.util import (
    lambda_name,
    ValidationError,
    NotUniqueError,
    NotFound,
    NameValidationError,
)
from meshed.itools import (
    topological_sort,
    leaf_nodes,
    root_nodes,
    descendants,
    ancestors,
)


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


from i2.signatures import ch_func_to_all_pk


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
    _func = ch_func_to_all_pk(func)  # makes a position-keyword copy of func
    output_key = output_name
    if output_name is None:
        output_key = _func.__name__

    def source_from_decorated():
        variables[output_key] = call_somewhat_forgivingly(_func, (), variables)

    return source_from_decorated


def _complete_dict_with_iterable_of_required_keys(
    to_complete: dict, complete_with: Iterable
):
    """Complete `to_complete` (in place) with `complete_with`
    `complete_with` contains values that must be covered by `to_complete`
    Those values that are not covered will be inserted in to_complete,
    with key=val

    >>> d = {'a': 'A', 'c': 'C'}
    >>> _complete_dict_with_iterable_of_required_keys(d, 'abc')
    >>> d
    {'a': 'A', 'c': 'C', 'b': 'b'}

    """
    keys_already_covered = set(to_complete)
    for required_key in complete_with:
        if required_key not in keys_already_covered:
            to_complete[required_key] = required_key


def _inverse_dict_asserting_losslessness(d: dict):
    inv_d = {v: k for k, v in d.items()}
    assert len(inv_d) == len(d), (
        f"can't invert: You have some duplicate values in this dict: " f'{d}'
    )
    return inv_d


def _extract_values(d: dict, keys: Iterable):
    """generator of values extracted from d for keys"""
    for k in keys:
        yield d[k]


def extract_values(d: dict, keys: Iterable):
    """Extract values from dict ``d``, returning them:

    - as a tuple if len(keys) > 1

    - a single value if len(keys) == 1

    - None if not

    This is used as the default extractor in DAG
    """
    tup = tuple(_extract_values(d, keys))
    if len(tup) > 1:
        return tup
    elif len(tup) == 1:
        return tup[0]
    else:
        return None


def extract_items(d: dict, keys: Iterable):
    """generator of (k, v) pairs extracted from d for keys"""
    for k in keys:
        yield k, d[k]


def _separate_func_nodes_and_var_nodes(nodes):
    func_nodes = list()
    var_nodes = list()
    for node in nodes:
        if is_func_node(node):
            func_nodes.append(node)
        else:
            var_nodes.append(node)
    return func_nodes, var_nodes


_not_found = object()


def _find_unique_element(item, search_iterable, key: Callable[[Any, Any], bool]):
    """Find item in search_iterable, using key as the matching function,
    raising a NotFound error if no match and a NotUniqueError if more than one."""
    it = filter(lambda x: key(item, x), search_iterable)
    first = next(it, _not_found)
    if first == _not_found:
        raise NotFound(f"{item} wasn't found")
    else:
        the_next_match = next(it, _not_found)
        if the_next_match is not _not_found:
            raise NotUniqueError(f"{item} wasn't unique")
    return first


ParameterMerger = Callable[[Iterable[Parameter]], Parameter]
conservative_parameter_merge: ParameterMerger


def conservative_parameter_merge(
    params, same_kind=True, same_default=True, same_annotation=True
):
    """Validates that all the params are exactly the same, returning the first if so.

    This is used when hooking up functions that use the same parameters (i.e. arg
    names). When the name of an argument is used more than once, which kind, default,
    and annotation should be used in the interface of the DAG?

    If they're all the same, there's no problem.

    But if they're not the same, we need to provide control on which to ignore.

    """
    suggestion_on_error = '''To resolve this you have several choices:
    
    - Change the properties of the param (kind, default, annotation) to be those you 
      want. For example, you can use ``i2.Sig.ch_param_attrs`` 
      (or ``i2.Sig.ch_defaults``, ``i2.Sig.ch_kinds``, ``i2.Sig.ch_annotations``)
      to get a function decorator that will do that for you.
    - If you're making a DAG, consider specifying a different ``parameter_merge``.
      For example you can use ``functools.partial`` on 
      ``i2.dag.conservative_parameter_merge``, fixing ``same_kind``, ``same_default``, 
      and/or ``same_annotation`` to ``False`` to get a more lenient version of it.
      
    See https://github.com/i2mint/meshed/issues/7 (description and comments) for more
    info.
    '''
    first_param, *_ = params
    if not all(p.name == first_param.name for p in params):
        raise ValidationError(
            f"Some params didn't have the same name: {params}\n{suggestion_on_error}"
        )
    if same_kind and not all(p.kind == first_param.kind for p in params):
        raise ValidationError(
            f"Some params didn't have the same kind: {params}\n{suggestion_on_error}"
        )
    if same_default and not all(p.default == first_param.default for p in params):
        raise ValidationError(
            f"Some params didn't have the same default: {params}\n{suggestion_on_error}"
        )
    if same_annotation and not all(
        p.annotation == first_param.annotation for p in params
    ):
        raise ValidationError(
            f"Some params didn't have the same annotation: "
            f'{params}\n{suggestion_on_error}'
        )
    return first_param


def modified_func_node(func_node, **modifications) -> FuncNode:
    modifiable_attrs = {'func', 'name', 'bind', 'out'}
    assert not modifications.keys().isdisjoint(
        modifiable_attrs
    ), f"Can only modify these: {', '.join(modifiable_attrs)}"
    original_func_node_kwargs = {
        'func': func_node.func,
        'name': func_node.name,
        'bind': func_node.bind,
        'out': func_node.out,
    }
    return FuncNode(**dict(original_func_node_kwargs, **modifications))


from i2 import partialx

# TODO: doctests
def partialized_funcnodes(func_nodes, **keyword_defaults):
    for func_node in func_nodes:
        if argnames_to_be_bound := set(keyword_defaults).intersection(
            func_node.sig.names
        ):
            bindings = dict(extract_items(keyword_defaults, argnames_to_be_bound))
            # partialize the func and move defaulted params to the end
            partialized_func = partialx(
                func_node.func, **bindings, _allow_reordering=True
            )
            # get rid of kinds  # TODO: This is a bit extreme -- consider gentler touch
            nice_kinds_sig = Sig(partialized_func).ch_kinds_to_position_or_keyword()
            nice_kinds_partialized_func = nice_kinds_sig(partialized_func)
            yield modified_func_node(
                func_node, func=nice_kinds_partialized_func
            )  # TODO: A better way without partial?
        else:
            yield func_node


Scope = dict
VarNames = Iterable[str]
DagOutput = Any


def _name_attr_or_x(x):
    return getattr(x, 'name', x)


# TODO: caching last scope isn't really the DAG's direct concern -- it's a debugging
#  concern. Perhaps a more general form would be to define a cache factory defaulting
#  to a dict, but that could be a "dict" that logs writes (even to an attribute of self)
@dataclass
class DAG:
    """
    >>> from meshed.dag import DAG, Sig
    >>>
    >>> def this(a, b=1):
    ...     return a + b
    >>> def that(x, b=1):
    ...     return x * b
    >>> def combine(this, that):
    ...     return (this, that)
    >>>
    >>> dag = DAG((this, that, combine))
    >>> print(dag.synopsis_string())
    x,b -> that_ -> that
    a,b -> this_ -> this
    this,that -> combine_ -> combine

    But what does it do?

    It's a callable, with a signature:

    >>> Sig(dag)  # doctest: +SKIP
    <Sig (x, a, b=1)>

    And when you call it, it executes the dag from the root values you give it and
    returns the leaf output values.

    >>> dag(1, 2, 3)  # (a+b,x*b) == (2+3,1*3) == (5, 3)
    (5, 3)
    >>> dag(1, 2)  # (a+b,x*b) == (2+1,1*1) == (3, 1)
    (3, 1)

    The above DAG was created straight from the functions, using only the names of the
    functions and their arguments to define how to hook the network up.

    But if you didn't write those functions specifically for that purpose, or you want
    to use someone else's functions, we got you covered.

    You can define the name of the node (the `name` argument), the name of the output
    (the `out` argument) and a mapping from the function's arguments names to
    "network names" (through the `bind` argument).
    The edges of the DAG are defined by matching `out` TO `bind`.

    """

    func_nodes: Iterable[Union[FuncNode, Callable]]
    cache_last_scope: bool = field(default=True, repr=False)
    parameter_merge: ParameterMerger = field(
        default=conservative_parameter_merge, repr=False
    )
    # can return a prepopulated scope too!
    new_scope: Callable = field(default=dict, repr=False)
    name: str = None
    extract_output_from_scope: Callable[[Scope, VarNames], DagOutput] = field(
        default=extract_values, repr=False
    )

    def __post_init__(self):
        self.func_nodes = tuple(_mk_func_nodes(self.func_nodes))
        self.graph = _func_nodes_to_graph_dict(self.func_nodes)
        self.nodes = topological_sort(self.graph)
        # reorder the nodes to fit topological order
        self.func_nodes, self.var_nodes = _separate_func_nodes_and_var_nodes(self.nodes)
        # self.sig = Sig(dict(extract_items(sig.parameters, 'xz')))
        self.sig = Sig(  # make a signature
            sort_params(  # with the sorted params (sorted to satisfy kind/default order)
                self.src_name_params(root_nodes(self.graph))
            )
        )
        self.sig(self)  # to put the signature on the callable DAG
        # figure out the roots and leaves
        self.roots = tuple(self.sig.names)  # roots in the same order as signature
        leafs = leaf_nodes(self.graph)
        # But we want leafs in topological order
        self.leafs = tuple([name for name in self.nodes if name in leafs])
        self.last_scope = None
        if self.name is not None:
            self.__name__ = self.name

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def _call(self, *args, **kwargs):
        # Get a dict of {argname: argval} pairs from positional and keyword arguments
        # How positionals are resolved is determined by sels.sig
        # The result is the initial ``scope`` the func nodes will both read from
        # to get their arguments, and write their outputs to.
        scope = self.sig.kwargs_from_args_and_kwargs(args, kwargs)
        # Go through self.func_nodes in order and call them on scope (performing said
        # read_input -> call_func -> write_output operations)
        self.call_on_scope(scope)
        # From the scope, that may contain all intermediary results,
        # extract the desired final output and return it
        return self.extract_output_from_scope(scope, self.leafs)

    def call_on_scope(self, scope=None):
        """Calls the func_nodes using scope (a dict or MutableMapping) both to
        source it's arguments and write it's results.

        Note: This method is only meant to be used as a backend to __call__, not as
        an actual interface method. Additional control/constraints on read and writes
        can be implemented by providing a custom scope for that. For example, one could
        log read and/or writes to specific keys, or disallow overwriting to an existing
        key (useful for pipeline sanity), etc.
        """
        if scope is None:
            scope = self.new_scope()  # fresh new scope
        if self.cache_last_scope:
            self.last_scope = scope  # just to remember it, for debugging purposes ONLY!

        for func_node in self.func_nodes:
            func_node.call_on_scope(scope)

    # def clone(self, *args, **kwargs):
    #     """Use args, kwargs to make an instance, using self attributes for
    #     unspecified arguments.
    #     """

    def __getitem__(self, item):
        """Get a sub-dag from a specification of (var or fun) input and output nodes.

        ``dag[input_nodes:output_nodes]`` is the sub-dag made of intersection of all
        descendants of ``input_nodes``
        (inclusive) and ancestors of ``output_nodes`` (inclusive), where additionally,
        when a func node is contained, it takes with it the input and output nodes
        it needs.

        >>> def f(a): ...
        >>> def g(f): ...
        >>> def h(g): ...
        >>> def i(h): ...
        >>> dag = DAG([f, g, h, i])

        See what this dag looks like (it's a simple pipeline):

        >>> dag = DAG([f, g, h, i])
        >>> print(dag.synopsis_string())
        a -> f_ -> f
        f -> g_ -> g
        g -> h_ -> h
        h -> i_ -> i

        Get a subdag from ``g_`` (indicates the function here) to the end of ``dag``

        >>> subdag = dag['g_':]
        >>> print(subdag.synopsis_string())
        f -> g_ -> g
        g -> h_ -> h
        h -> i_ -> i

        From the beginning to ``h_``

        >>> print(dag[:'h_'].synopsis_string())
        a -> f_ -> f
        f -> g_ -> g
        g -> h_ -> h

        From ``g_`` to ``h_`` (both inclusive)

        >>> print(dag['g_':'h_'].synopsis_string())
        f -> g_ -> g
        g -> h_ -> h

        Above we used function (node names) to specify what we wanted, but we can also
        use names of input/output var-nodes. Do note the difference though.
        The nodes you specify to get a sub-dag are INCLUSIVE, but when you
        specify function nodes, you also get the input and output nodes of these
        functions.

        The ``dag['g_', 'h_']`` give us a sub-dag starting at ``f`` (the input node),
        but when we ask ``dag['g', 'h_']`` instead, ``g`` being the output node of
        function node ``g_``, we only get ``g -> h_ -> h``:

        >>> print(dag['g':'h'].synopsis_string())
        g -> h_ -> h

        If we wanted to include ``f`` we'd have to specify it:

        >>> print(dag['f':'h'].synopsis_string())
        f -> g_ -> g
        g -> h_ -> h

        Those were for simple pipelines, but let's now look at a more complex dag.

        We'll let the following examples self-comment:

        >>> def f(u, v): ...
        >>> def g(f): ...
        >>> def h(f, w): ...
        >>> def i(g, h): ...
        >>> def j(h, x): ...
        >>> def k(i): ...
        >>> def l(i, j): ...
        >>> dag = DAG([f, g, h, i, j, k, l])
        >>> print(dag.synopsis_string())
        u,v -> f_ -> f
        f,w -> h_ -> h
        h,x -> j_ -> j
        f -> g_ -> g
        g,h -> i_ -> i
        i,j -> l_ -> l
        i -> k_ -> k

        A little util to get consistent prints:

        >>> def print_sorted_synopsis(dag):
        ...     t = sorted(dag.synopsis_string().split('\\n'))
        ...     print('\\n'.join(t))

        >>> print_sorted_synopsis(dag[['u', 'f']:'h'])
        f,w -> h_ -> h
        u,v -> f_ -> f
        >>> print_sorted_synopsis(dag['u':'h'])
        f,w -> h_ -> h
        u,v -> f_ -> f
        >>> print_sorted_synopsis(dag[['u', 'f']:['h', 'g']])
        f -> g_ -> g
        f,w -> h_ -> h
        u,v -> f_ -> f
        >>> print_sorted_synopsis(dag[['x', 'g']:'k'])
        g,h -> i_ -> i
        i -> k_ -> k
        >>> print_sorted_synopsis(dag[['x', 'g']:['l', 'k']])
        g,h -> i_ -> i
        h,x -> j_ -> j
        i -> k_ -> k
        i,j -> l_ -> l

        >>>

        """
        return self._getitem(item)

    def _getitem(self, item):
        ins, outs = self.process_item(item)
        _descendants = set(
            filter(FuncNode.has_as_instance, set(ins) | descendants(self.graph, ins))
        )
        _ancestors = set(
            filter(FuncNode.has_as_instance, set(outs) | ancestors(self.graph, outs))
        )
        subgraph_nodes = _descendants.intersection(_ancestors)
        # TODO: When clone ready, use to do `constructor = type(self)` instead of DAG
        # constructor = type(self)  # instead of DAG
        return DAG(
            func_nodes=subgraph_nodes,
            cache_last_scope=self.cache_last_scope,
            parameter_merge=self.parameter_merge,
        )

    # TODO: Think about adding a ``_roll_in_orphaned_nodes=False`` argument:
    #   See https://github.com/i2mint/meshed/issues/14 for more information.
    def partial(
        self,
        *positional_dflts,
        _remove_bound_arguments=False,
        _consider_defaulted_arguments_as_bound=False,
        **keyword_dflts,
    ):
        """Get a curried version of the DAG.

        Like ``functools.partial``, but returns a DAG (not just a callable) and allows
        you to remove bound arguments as well as roll in orphaned_nodes.

        :param positional_dflts: Bind arguments positionally
        :param keyword_dflts: Bind arguments through their names
        :param _remove_bound_arguments: False -- set to True if you don't want bound
            arguments to show up in the signature.
        :param _consider_defaulted_arguments_as_bound: False -- set to True if
            you want to also consider arguments that already had defaults as bound
            (and be removed).
        :return:

        >>> def f(a, b):
        ...     return a + b
        >>> def g(c, d=4):
        ...     return c * d
        >>> def h(f, g):
        ...     return g - f
        >>> dag = DAG([f, g, h])
        >>> from inspect import signature
        >>> str(signature(dag))
        '(c, a, b, d=4)'
        >>> dag(3, 1, 2, 4)  # == (3 * 4) - (1 + 2) == 12 - 3 == 9
        9
        >>> dag(c=3, a=1, b=2, d=4)  # same as above
        9

        >>> new_dag = dag.partial(c=3)
        >>> isinstance(new_dag, DAG)  # it's a dag (not just a partialized callable!)
        True
        >>> str(signature(new_dag))
        '(a, b, c=3, d=4)'
        >>> new_dag(1, 2)  # same as dag(c=3, a=1, b=2, d=4), so:
        9
        """

        keyword_dflts = self.sig.kwargs_from_args_and_kwargs(
            args=positional_dflts,
            kwargs=keyword_dflts,
            apply_defaults=_consider_defaulted_arguments_as_bound,
            # positional_dflts and keyword_dflts usually don't cover all arguments, so:
            allow_partial=True,
            # we prefer to let the user know if they're trying to bind arguments
            # that don't exist, so:
            allow_excess=False,
            # we don't really care about kind, so:
            ignore_kind=True,
        )
        # TODO: mk_instance: What about other init args (cache_last_scope, ...)?
        mk_instance = type(self)
        func_nodes = partialized_funcnodes(self, **keyword_dflts)
        new_dag = mk_instance(func_nodes)
        if _remove_bound_arguments:
            new_sig = Sig(new_dag).remove_names(list(keyword_dflts))
            new_sig(new_dag)  # Change the signature of new_dag with bound args removed
        return new_dag

    def process_item(self, item):
        assert isinstance(item, slice), f'must be a slice, was: {item}'

        input_names, outs = item.start, item.stop

        empty_slice = slice(None)

        def ensure_variable_list(obj):
            if obj is None:
                return self.var_nodes
            if isinstance(obj, str):
                obj = obj.split()
            if isinstance(obj, (str, Callable)):
                return [self.get_node_matching(obj)]
            elif isinstance(obj, Iterable):
                return list(map(self.get_node_matching, obj))
            else:
                raise ValidationError(f'Unrecognized variables specification: {obj}')

        # assert len(item) == 2, f"Only items of size 1 or 2 are supported"
        input_names, outs = map(ensure_variable_list, [input_names, outs])
        return input_names, outs

    def get_node_matching(self, pattern):
        if isinstance(pattern, str):
            if pattern in self.var_nodes:
                return pattern
            return self.func_node_for_name(pattern)
        elif isinstance(pattern, Callable):
            return self.func_node_for_func(pattern)
        raise NotFound(f'No matching node: {pattern}')

    def func_node_for_name(self, name):
        return _find_unique_element(
            name, self.func_nodes, lambda name, fn: name == fn.name
        )

    def func_node_for_func(self, func):
        return _find_unique_element(
            func, self.func_nodes, lambda func, fn: func == fn.func
        )

    def __iter__(self):
        """Yields the self.func_nodes
        Note: The raison d'etre of this ``__iter__`` is simply because if no custom one
        is provided, python defaults to yielding ``__getitem__[i]`` for integers,
        which leads to an error being raised.

        At least here we yield something sensible.

        A consequence of the `__iter__` being the iterable of func_nodes is that we
        can extend dags using the star operator. Consider the following dag;

        >>> def f(a): return a * 2
        >>> def g(f, b): return f + b
        >>> dag = DAG([f, g])
        >>> assert dag(2, 3) == 7

        Say you wanted to now take a, b, and the output og g, and feed it to another
        function...

        >>> def h(a, b, g): return f"{a=} {b=} {g=}"
        >>> extended_dag = DAG([*dag, h])
        >>> extended_dag(a=2, b=3)
        'a=2 b=3 g=7'
        """
        yield from self.func_nodes

    # ------------ utils --------------------------------------------------------------

    @property
    def params_for_src(self):
        """The ``{src_name: list_of_params_using_that_src,...}`` dictionary.
        That is, a ``dict`` having lists of all ``Parameter`` objs that are used by a
        ``node.bind`` source (value of ``node.bind``) for each such source in the graph

        For each ``func_node``, ``func_node.bind`` gives us the
        ``{param: varnode_src_name}`` specification that tells us where (key of scope)
        to source the arguments of the ``func_node.func`` for each ``param`` of that
        function.

        What ``params_for_src`` is, is the corresponding inverse map.
        The ``{varnode_src_name: list_of_params}`` gathered by scanning each
        ``func_node`` of the DAG.
        """
        d = defaultdict(list)
        for node in self.func_nodes:
            for arg_name, src_name in node.bind.items():
                d[src_name].append(node.sig.parameters[arg_name])
        return dict(d)

    def src_name_params(self, src_names: Optional[Iterable[str]] = None):
        # see params_for_src property to see what d is
        d = self.params_for_src
        if src_names is None:  # if no src_names given, use the names of all var_nodes
            src_names = set(d)

        # For every src_name of the DAG that is in ``src_name``...
        for src_name in filter(src_names.__contains__, d):
            params = d[src_name]  # consider all the params that use it
            # make version of these params that have the same name (namely src_name)
            params_with_name_changed_to_src_name = [
                p.replace(name=src_name) for p in params
            ]
            if len(params_with_name_changed_to_src_name) == 1:
                # if there's only one param, return it (there can be no conflict)
                yield params_with_name_changed_to_src_name[0]
            else:  # if there's more than one param, merge them
                # How to resolve conflicts (different defaults, annotations or kinds)
                # is determined by what ``parameter_merge`` specified, which is,
                # by default, strict (everything needs to be the same, or
                # ``parameter_merge`` with raise an error.
                yield self.parameter_merge(params_with_name_changed_to_src_name)

    @property
    def graph_ids(self):
        """The dict representing the ``{from_node: to_nodes}`` graph.
        Like ``.graph``, but with node ids (names).
        """
        return {
            _name_attr_or_x(k): list(map(_name_attr_or_x, v))
            for k, v in self.graph.items()
        }

    # ------------ display --------------------------------------------------------------

    def synopsis_string(self):
        return '\n'.join(func_node.synopsis_string() for func_node in self.func_nodes)

    # TODO: Give more control (merge with lined)
    def dot_digraph_body(self, start_lines=()):
        """Make lines for dot (graphviz) specification of DAG

        >>> def add(a, b=1): return a + b
        >>> def mult(x, y=3): return x * y
        >>> def exp(mult, a): return mult ** a
        >>> func_nodes = [
        ...     FuncNode(add, out='x'), FuncNode(mult, name='the_product'), FuncNode(exp)
        ... ]

        #
        # >>> assert list(DAG(func_nodes).dot_digraph_body()) == [
        # ]
        """
        yield from dot_lines_of_func_nodes(self.func_nodes, start_lines=start_lines)

    @wraps(dot_digraph_body)
    def dot_digraph_ascii(self, *args, **kwargs):
        """Get an ascii art string that represents the pipeline"""
        from meshed.util import dot_to_ascii

        return dot_to_ascii('\n'.join(self.dot_digraph_body(*args, **kwargs)))

    @wraps(dot_digraph_body)
    def dot_digraph(self, *args, **kwargs):
        try:
            import graphviz
        except (ModuleNotFoundError, ImportError) as e:
            raise ModuleNotFoundError(
                f'{e}\nYou may not have graphviz installed. '
                f'See https://pypi.org/project/graphviz/.'
            )
        # Note: Since graphviz 0.18, need to have a newline in body lines!
        body = list(map(_add_new_line_if_none, self.dot_digraph_body(*args, **kwargs)))
        return graphviz.Digraph(body=body)


# These are the defaults used in lined.
# TODO: Merge some of the functionalities around graph displays in lined and meshed
dflt_configs = dict(
    fnode_shape='box',
    vnode_shape='none',
    display_all_arguments=True,
    edge_kind='to_args_on_edge',
    input_node=True,
    output_node='output',
)


def param_to_dot_definition(p: Parameter, shape=dflt_configs['vnode_shape']):
    if p.default is not empty:
        name = p.name + '='
    elif p.kind == p.VAR_POSITIONAL:
        name = '*' + p.name
    elif p.kind == p.VAR_KEYWORD:
        name = '**' + p.name
    else:
        name = p.name
    yield f'{p.name} [label="{name}" shape="{shape}"]'


def call_func(func, kwargs):
    kwargs = {k.__name__: v for k, v in kwargs.items()}
    return Sig(func).source_kwargs(kwargs)


def dot_lines_of_func_parameters(
    parameters: Iterable[Parameter],
    out: str,
    func_id: str,
    *,
    func_label: str = None,
    output_shape: str = dflt_configs['vnode_shape'],
    func_shape: str = dflt_configs['fnode_shape'],
) -> Iterable[str]:
    assert func_id != out, (
        f"Your func and output name shouldn't be the " f'same: {out=} {func_id=}'
    )
    func_label = func_label or func_id
    yield f'{out} [label="{out}" shape="{output_shape}"]'
    yield f'{func_id} [label="{func_label}" shape="{func_shape}"]'
    yield f'{func_id} -> {out}'
    # args -> func
    for p in parameters:
        yield from param_to_dot_definition(p)
    for p in parameters:
        yield f'{p.name} -> {func_id}'


def _parameters_and_names_from_sig(
    sig: Sig, out=None, func_name=None,
):
    func_name = func_name or sig.name
    out = out or sig.name
    if func_name == out:
        func_name = '_' + func_name
    assert isinstance(func_name, str) and isinstance(out, str)
    return sig.parameters, out, func_name


def dot_lines_of_func_nodes(func_nodes: Iterable[FuncNode], start_lines=()):
    r"""Got lines generator for the graphviz.DiGraph(body=list(...))

    >>> def add(a, b=1):
    ...     return a + b
    >>> def mult(x, y=3):
    ...     return x * y
    >>> def exp(mult, a):
    ...     return mult ** a
    >>> func_nodes = [
    ...     FuncNode(add, out='x'),
    ...     FuncNode(mult, name='the_product'),
    ...     FuncNode(exp)
    ... ]
    >>> lines = list(dot_lines_of_func_nodes(func_nodes))
    >>> assert lines == [
    ... 'x [label="x" shape="none"]',
    ... '_add [label="_add" shape="box"]',
    ... '_add -> x',
    ... 'a [label="a" shape="none"]',
    ... 'b [label="b=" shape="none"]',
    ... 'a -> _add',
    ... 'b -> _add',
    ... 'mult [label="mult" shape="none"]',
    ... 'the_product [label="the_product" shape="box"]',
    ... 'the_product -> mult',
    ... 'x [label="x" shape="none"]',
    ... 'y [label="y=" shape="none"]',
    ... 'x -> the_product',
    ... 'y -> the_product',
    ... 'exp [label="exp" shape="none"]',
    ... '_exp [label="_exp" shape="box"]',
    ... '_exp -> exp',
    ... 'mult [label="mult" shape="none"]',
    ... 'a [label="a" shape="none"]',
    ... 'mult -> _exp',
    ... 'a -> _exp'
    ... ]  # doctest: +SKIP

    >>> from meshed.util import dot_to_ascii
    >>>
    >>> print(dot_to_ascii('\n'.join(lines)))  # doctest: +SKIP
    <BLANKLINE>
                    a        ─┐
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
     b=  ──▶ │    _add     │  │
             └─────────────┘  │
               │              │
               │              │
               ▼              │
                              │
                    x         │
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
     y=  ──▶ │ the_product │  │
             └─────────────┘  │
               │              │
               │              │
               ▼              │
                              │
                  mult        │
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
             │    _exp     │ ◀┘
             └─────────────┘
               │
               │
               ▼
    <BLANKLINE>
                   exp
    <BLANKLINE>

    """
    yield from start_lines
    validate_that_func_node_names_are_sane(func_nodes)
    for func_node in func_nodes:
        yield from dot_lines_of_func_node(func_node)


def dot_lines_of_func_node(func_node: FuncNode):

    out = func_node.out
    func_id = func_node.name
    func_label = getattr(func_node, 'func_label', func_id)
    if out == func_id:  # though forbidden in default FuncNode validation
        func_id = '_' + func_id

    # Get the Parameter objects for sig, with names changed to bind ones
    params = func_node.sig.ch_names(**func_node.bind).params

    yield from dot_lines_of_func_parameters(
        params, out=out, func_id=func_id, func_label=func_label
    )


def _add_new_line_if_none(s: str):
    """Since graphviz 0.18, need to have a newline in body lines.
    This util is there to address that, adding newlines to body lines
    when missing."""
    if s and s[-1] != '\n':
        return s + '\n'
    return s


# ---------- with ext.gk -------------------------------------------------------

with suppress(ModuleNotFoundError, ImportError):
    from meshed.ext.gk import operation, Operation

    def funcs_to_operations(*funcs, exclude_names=()) -> Operation:
        """Get an operation from a callable"""

        for func in funcs:
            _func_name = mk_func_name(func, exclude_names)
            exclude_names = exclude_names + (_func_name,)
            needs = arg_names(func, _func_name, exclude_names)
            exclude_names = exclude_names + tuple(needs)
            yield operation(
                func, name=_func_name, needs=needs, provides=_func_name,
            )

    def funcs_to_operators(*funcs, exclude_names=()) -> Operation:
        """Get an operation from a callable"""

        for func, operation in zip(funcs, funcs_to_operations(funcs, exclude_names)):
            yield operation(func)
