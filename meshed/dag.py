"""
Making DAGs
"""

from contextlib import suppress
from typing import Callable
from functools import partial, wraps
from collections import Counter
from i2 import Sig

from meshed.util import name_of_obj
from meshed.itools import topological_sort, add_edge, leaf_nodes, root_nodes


def find_first_free_name(prefix, exclude_names=(), start_at=2):
    if prefix not in exclude_names:
        return prefix
    else:
        i = start_at
        while True:
            name = f"{prefix}__{i}"
            if name not in exclude_names:
                return name
            i += 1


def func_name(func, exclude_names=()):
    name = getattr(func, "__name__", "")
    return find_first_free_name(name, exclude_names)


def arg_names(func, func_name, exclude_names=()):
    names = Sig(func).names

    def gen():
        _exclude_names = exclude_names
        for name in names:
            if name not in _exclude_names:
                yield name
            else:
                found_name = find_first_free_name(f"{func_name}__{name}", _exclude_names)
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


from dataclasses import dataclass, field
from typing import Callable, MutableMapping, Mapping, Optional, Iterable

from i2.signatures import call_forgivingly, call_somewhat_forgivingly
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
    _func = ch_func_to_all_pk(func)  # makes a position-keyword copy of func
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


def _complete_dict_with_iterable_of_required_keys(
    to_complete: dict, complete_with: Iterable
):
    """Complete `to_complete` (in place) with `complete_with`
    `complete_with` contains values that must be covered by `to_complete`
    Those values that are not covered will be inserted in to_complete,
    with key=val

    >>> d = {'a': 'A', 'c': 'C'}
    >>> _complete_dict_with_iterable_when_values_missing(d, 'ABC')
    >>> d
    {'a': 'A', 'c': 'C', 'B': 'B'}

    """
    keys_already_covered = set(to_complete)
    for required_key in complete_with:
        if required_key not in keys_already_covered:
            to_complete[required_key] = required_key


def _inverse_dict_asserting_losslessness(d: dict):
    inv_d = {v: k for k, v in d.items()}
    assert len(inv_d) == len(d), (
        f"can't invert: You have some duplicate values in this dict: " f"{d}"
    )
    return inv_d


def _mapped_extraction(extract_from: dict, key_map: dict):
    """for every (k, v) of key_map whose v is a key of extract_from, yields
    (v, extract_from[v])

    Meant to be curried into an extractor, and wrapped in dict.

    >>> extracted = _mapped_extraction(
    ...     {'a': 1, 'b': 2, 'c': 3}, # extract_from
    ...     {'A': 'a', 'C': 'c', 'D': 'd'}  # note that there's no 'd' in extract_from
    ... )
    >>> dict(extracted)
    {'a': 1, 'c': 3}

    """
    for k, v in key_map.items():
        if v in extract_from:
            yield v, extract_from[v]


# TODO: Think of the hash more carefully.
@dataclass
class FuncNode:
    """
    :param func: Function to wrap
    :src_names: The {func_argname: external_name,...} mapping that defines where
        the node will source the data to call the function.
        This only has to be used if the external names are different from the names
        of the arguments of the function.


    """

    func: Callable
    name: str = field(default=None)
    src_names: dict = field(default_factory=dict)
    return_name: str = field(default=None)
    write_output_into_src_kwargs = (True,)
    _name_prefix: str = field(default="_", init=False, repr=False)
    _return_name_suffix: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        self.name = self.name or self._name_prefix + name_of_obj(self.func)
        self.return_name = (
            self.return_name or name_of_obj(self.func) + self._return_name_suffix
        )
        # TODO: assert name != return_name?

        sig = Sig(self.func)
        # complete src_names with the argnames of the signature
        src_names_not_in_sig_names = self.src_names.keys() - sig.names
        assert not src_names_not_in_sig_names, (
            f"some src_names keys weren't found as function argnames: "
            f"{', '.join(src_names_not_in_sig_names)}"
        )
        _complete_dict_with_iterable_of_required_keys(self.src_names, sig.names)
        self.extractor = partial(_mapped_extraction, key_map=self.src_names)
        self.func_sig = sig

        # # TODO: Should we changed the sig to match the source? Using Sig.ch_param_attrs
        # sig(self)  # puts the signature of func on the call of self

    def call_on_scope(self, scope: dict):
        relevant_kwargs = dict(self.extractor(scope))
        # print(scope, relevant_kwargs)
        output = call_somewhat_forgivingly(
            self.func, (), relevant_kwargs, enforce_sig=self.func_sig
        )
        if self.write_output_into_src_kwargs:
            scope[self.return_name] = output
        return output

    def _hash_str(self):
        ",".join(self.src_names) + " -> " + self.return_name

    # TODO: Find a better one
    def __hash__(self):
        return id(self)

    def __call__(self, scope):
        """Deprecated: Don't use. Might be a normal function with a signature"""
        return self.call_on_scope(scope)


def validate_that_func_node_names_are_sane(func_nodes: Iterable[FuncNode]):
    """Assert that the names of func_nodes are sane.
    That is:
        - are valid dot (graphviz) names (we'll use str.isidentifier because lazy)
        - All the func.name and func.return_name are unique
        - more to come
    """
    func_nodes = list(func_nodes)
    node_names = [x.name for x in func_nodes]
    return_names = [x.return_name for x in func_nodes]
    assert all(
        map(str.isidentifier, node_names)
    ), f"some node names weren't valid identifiers: {node_names}"
    assert all(
        map(str.isidentifier, return_names)
    ), f"some return names weren't valid identifiers: {return_names}"
    if len(set(node_names) | set(return_names)) != 2 * len(func_nodes):
        c = Counter(node_names + return_names)
        offending_names = [name for name, count in c.items() if count > 1]
        raise ValueError(
            f"Some of your node names and/or return_names where used more than once. "
            f"They shouldn't. These are the names I find offensive: {offending_names}"
        )


from meshed.makers import edge_reversed_graph


def _mk_func_nodes(func_nodes):
    # TODO: Take care of names (or track and take care if collision)
    for func_node in func_nodes:
        if isinstance(func_node, FuncNode):
            yield func_node
        elif isinstance(func_node, Callable):
            yield FuncNode(func_node)
        else:
            raise TypeError(f"Can't convert this to a FuncNode: {func_node}")


def _func_nodes_to_graph_dict(func_nodes):
    g = dict()

    for f in func_nodes:
        for src_name in f.src_names.values():
            add_edge(g, src_name, f)
        add_edge(g, f, f.return_name)
    return g


def _is_func_node(obj) -> bool:
    return isinstance(obj, FuncNode)


def _is_not_func_node(obj) -> bool:
    return not _is_func_node(obj)


def _extract_keys(d: dict, keys: Iterable):
    for k in keys:
        yield d[k]


# TODO: Make a __getitem__that returns a function with specific args and return vals
#   f = dag['a', 'b'] would be a callable (still a dag?) with args a and b in that order
#   f['x', 'y'] would be like f, except returning the tuple (x, y) instead of the
#   whole thing.
@dataclass
class DAG:
    func_nodes: Iterable[FuncNode]

    def __post_init__(self):
        self.func_nodes = tuple(_mk_func_nodes(self.func_nodes))
        self.graph = _func_nodes_to_graph_dict(self.func_nodes)
        self.nodes = topological_sort(self.graph)
        # reorder the nodes to fit topplogical order
        self.func_nodes = tuple(filter(_is_func_node, self.nodes))
        # figure out the roots and leaves
        self.roots = set(root_nodes(self.graph))
        self.leafs = set(leaf_nodes(self.graph))
        self.sig = Sig(self.roots)  # TODO: add defaults
        self.sig(self)  # to put the signature on the callable DAG

    def __call__(self, *args, **kwargs):
        scope = self.sig.kwargs_from_args_and_kwargs(args, kwargs)
        self.call_on_scope(scope)
        tup = tuple(_extract_keys(scope, self.leafs))
        if len(tup) > 1:
            return tup
        elif len(tup) == 1:
            return tup[0]
        else:
            return None

    def call_on_scope(self, scope=None):
        if scope is None:
            scope = dict()  # fresh new scope
        self._last_scope = scope  # just to remember it, for debugging purposes ONLY!

        for func_node in self.func_nodes:
            func_node.call_on_scope(scope)

    # TODO: Give more control (merge with lined)
    def dot_digraph_body(self):
        yield from dot_lines_of_func_nodes(self.func_nodes)

    @wraps(dot_digraph_body)
    def dot_digraph_ascii(self, *args, **kwargs):
        """Get an ascii art string that represents the pipeline"""
        from lined.util import dot_to_ascii

        return dot_to_ascii("\n".join(self.dot_digraph_body(*args, **kwargs)))

    @wraps(dot_digraph_body)
    def dot_digraph(self, *args, **kwargs):
        try:
            import graphviz
        except (ModuleNotFoundError, ImportError) as e:
            raise ModuleNotFoundError(
                f"{e}\nYou may not have graphviz installed. "
                f"See https://pypi.org/project/graphviz/."
            )

        body = list(self.dot_digraph_body(*args, **kwargs))
        return graphviz.Digraph(body=body)


from typing import Iterable
from i2.signatures import Parameter, empty, Sig


# These are the defaults used in lined.
# TODO: Merge some of the functionalities around graph displays in lined and meshed
dflt_configs = dict(
    fnode_shape="box",
    vnode_shape="none",
    display_all_arguments=True,
    edge_kind="to_args_on_edge",
    input_node=True,
    output_node="output",
)


def param_to_dot_definition(p: Parameter, shape=dflt_configs["vnode_shape"]):
    if p.default is not empty:
        name = p.name + "="
    else:
        name = p.name
    yield f'{p.name} [label="{name}" shape="{shape}"]'


def call_func(func, kwargs):
    kwargs = {k.__name__: v for k, v in kwargs.items()}
    return Sig(func).source_kwargs(kwargs)


def dot_lines_of_func_parameters(
    parameters: Iterable[Parameter],
    output_name: str,
    func_name: str,
    output_shape: str = dflt_configs["vnode_shape"],
    func_shape: str = dflt_configs["fnode_shape"],
) -> Iterable[str]:
    assert func_name != output_name, (
        f"Your func and output name shouldn't be the "
        f"same: {output_name=} {func_name=}"
    )
    yield f'{output_name} [label="{output_name}" shape="{output_shape}"]'
    yield f'{func_name} [label="{func_name}" shape="{func_shape}"]'
    yield f"{func_name} -> {output_name}"
    # args -> func
    for p in parameters.values():
        yield from param_to_dot_definition(p)
    for argname in parameters:
        yield f"{argname} -> {func_name}"


def _parameters_and_names_from_sig(
    sig: Sig,
    output_name=None,
    func_name=None,
):
    func_name = func_name or sig.name
    output_name = output_name or sig.name
    if func_name == output_name:
        func_name = "_" + func_name
    assert isinstance(func_name, str) and isinstance(output_name, str)
    return sig.parameters, output_name, func_name


def dot_lines_of_func_nodes(func_nodes: Iterable[FuncNode]):
    r"""Got lines generator for the graphviz.DiGraph(body=list(...))

    >>> def add(a, b=1):
    ...     return a + b
    >>> def mult(x, y=3):
    ...     return x * y
    >>> def exp(mult, a):
    ...     return mult ** a
    >>> func_nodes = [
    ...     FuncNode(add, return_name='x'),
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
    ... ]

    >>> from lined.util import dot_to_ascii
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
    validate_that_func_node_names_are_sane(func_nodes)
    for func_node in func_nodes:
        sig = func_node.func_sig
        output_name = func_node.return_name
        func_name = func_node.name
        if output_name == func_name:
            func_name = "_" + func_name
        yield from dot_lines_of_func_parameters(
            sig.parameters,
            output_name=output_name,
            func_name=func_name,
        )


# ---------- with ext.gk -------------------------------------------------------

with suppress(ModuleNotFoundError, ImportError):
    from meshed.ext.gk import operation, Network, Operation

    def funcs_to_operations(*funcs, exclude_names=()) -> Operation:
        """Get an operation from a callable"""

        for func in funcs:
            _func_name = func_name(func, exclude_names)
            exclude_names = exclude_names + (_func_name,)
            needs = arg_names(func, _func_name, exclude_names)
            exclude_names = exclude_names + tuple(needs)
            yield operation(
                func,
                name=_func_name,
                needs=needs,
                provides=_func_name,
            )

    def funcs_to_operators(*funcs, exclude_names=()) -> Operation:
        """Get an operation from a callable"""

        for func, operation in zip(funcs, funcs_to_operations(funcs, exclude_names)):
            yield operation(func)
