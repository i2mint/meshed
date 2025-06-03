"""
Base functionality of meshed
"""

from collections import Counter
from dataclasses import dataclass, field, fields
from functools import partial, cached_property
from typing import Callable, MutableMapping, Iterable, Union, Sized, Sequence, Literal

from i2 import Sig, call_somewhat_forgivingly
from i2.signatures import (
    ch_variadics_to_non_variadic_kind,
    CallableComparator,
    compare_signatures,
)
from meshed.util import ValidationError, NameValidationError, mk_func_name
from meshed.itools import add_edge

BindInfo = Literal["var_nodes", "params", "hybrid"]


def underscore_func_node_names_maker(func: Callable, name=None, out=None):
    """This name maker will resolve names in the following fashion:

     #. look at the (func) name and out given as arguments, if None...
     #. use mk_func_name(func) to make names.

    It will use the mk_func_name(func)  itself for out, but suffix the same with
    an underscore to provide a mk_func_name.

    This is so because here we want to allow easy construction of function networks
    where a function's output will be used as another's input argument when
    that argument has the the function's (output) name.
    """
    if out is None and hasattr(func, "_provides"):
        if len(func._provides) > 0:
            out = func._provides[0]
    if name is not None and out is not None:
        if name == out:
            name = name + "_"
        return name, out

    try:
        name_of_func = mk_func_name(func)
    except NameValidationError as err:
        err_msg = err.args[0]
        err_msg += (
            f"\nSuggestion: You might want to specify a name explicitly in "
            f"FuncNode(func, name=name) instead of just giving me the func as is."
        )
        raise NameValidationError(err_msg)
    if name is None and out is None:
        return name_of_func + "_", name_of_func
    elif out is None:
        return name, "_" + name
    elif name is None:
        if name_of_func == out:
            name_of_func += "_"
        return name_of_func, out


def basic_node_validator(func_node):
    """Validates a func node. Raises ValidationError if something wrong. Returns None.

    Validates:

    * that the ``func_node`` params are valid, that is, if not ``None``
        * ``func`` should be a callable
        * ``name`` and ``out`` should be ``str``
        * ``bind`` should be a ``Dict[str, str]``
    * that the names (``.name``, ``.out`` and all ``.bind.values()``)
        * are valid python identifiers (alphanumeric or underscore not starting with
          digit)
        * are not repeated (no duplicates)
    * that ``.bind.keys()`` are indeed present as params of ``.func``

    """
    _func_node_args_validation(
        func=func_node.func, name=func_node.name, bind=func_node.bind, out=func_node.out
    )
    names = [func_node.name, func_node.out, *func_node.bind.values()]

    names_that_are_not_strings = [name for name in names if not isinstance(name, str)]
    if names_that_are_not_strings:
        names_that_are_not_strings = ", ".join(map(str, names_that_are_not_strings))
        raise ValidationError(f"Should be strings: {names_that_are_not_strings}")

    # Make sure there's no name duplicates
    _duplicates = duplicates(names)
    if _duplicates:
        raise ValidationError(f"{func_node} has duplicate names: {_duplicates}")

    # Make sure all names are identifiers
    _non_identifiers = list(filter(lambda name: not name.isidentifier(), names))
    # print(_non_identifiers, names)
    if _non_identifiers:
        raise ValidationError(f"{func_node} non-identifier names: {_non_identifiers}")

    # Making sure all src_name keys are in the function's signature
    bind_names_not_in_sig_names = func_node.bind.keys() - func_node.sig.names
    assert not bind_names_not_in_sig_names, (
        f"some bind keys weren't found as function argnames: "
        f"{', '.join(bind_names_not_in_sig_names)}"
    )


def handle_variadics(func):
    func = ch_variadics_to_non_variadic_kind(func)
    # sig = Sig(func)
    # var_kw = sig.var_keyword_name

    #   # may be always return the wrapped function
    # # func.var_kw_name = var_kw # TODO add it when needed

    return func


# TODO: When 3.10, look into and possibly use match_args in to_dict and from_dict
# TODO: Make FuncNode immutable (is there a way to use frozen=True with post_init?)
# TODO: How to get a safe hash? Needs to be immutable only?
# TODO: FuncNode(func_node) gives us FuncNode(scope -> ...). Should we have it be
#  FuncNode.from_dict(func_node.to_dict()) instead?
# @dataclass(eq=True, order=True, unsafe_hash=True)
@dataclass(order=True)
class FuncNode:
    """A function wrapper that makes the function amenable to operating in a network.

    :param func: Function to wrap
    :param name: The name to associate to the function
    :param bind: The {func_argname: external_name,...} mapping that defines where
        the node will source the data to call the function.
        This only has to be used if the external names are different from the names
        of the arguments of the function.
    :param out: The variable name the function should write it's result to

    Like we stated: `FuncNode` is meant to operate in computational networks.
    But knowing what it does will help you make the networks you want, so we commend
    your curiousity, and will oblige with an explanation.

    Say you have a function to multiply numbers.

    >>> def multiply(x, y):
    ...     return x * y

    And you use it in some code like this:

    >>> item_price = 3.5
    >>> num_of_items = 2
    >>> total_price = multiply(item_price, num_of_items)

    What the execution of `total_price = multiply(item_price, num_of_items)` does is
    - grab the values (in the locals scope -- a dict), of ``item_price`` and ``num_of_items``,
    - call the multiply function on these, and then
    - write the result to a variable (in locals) named ``total_price``

    `FuncNode` is a function wrapper that specification of such a
    `output = function(...inputs...)` assignment statement
    in such a way that it can carry it out on a `scope`.
    A `scope` is a `dict` where the function can find it's input values and write its
    output values.

    For example, the `FuncNode` form of the above statement would be:

    >>> func_node = FuncNode(
    ...     func=multiply,
    ...     bind={'x': 'item_price', 'y': 'num_of_items'})
    >>> func_node
    FuncNode(x=item_price,y=num_of_items -> multiply_ -> multiply)

    Note the `bind` is a mapping **from** the variable names of the wrapped function
    **to** the names of the scope.

    That is, when it's time to execute, it tells the `FuncNode` where to find the values
    of its inputs.

    If an input is not specified in this `bind` mapping, the scope
    (external) name is supposed to be the same as the function's (internal) name.

    The purpose of a `FuncNode` is to source some inputs somewhere, compute something
    with these, and write the result somewhere. That somewhere is what we call a
    scope. A scope is a dictionary (or any mutuable mapping to be precise) and it works
    like this:

    >>> scope = {'item_price': 3.5, 'num_of_items': 2}
    >>> func_node.call_on_scope(scope)  # see that it returns 7.0
    7.0
    >>> scope  # but also wrote this in the scope
    {'item_price': 3.5, 'num_of_items': 2, 'multiply': 7.0}

    Consider ``item_price,num_of_items -> multiply_ -> multiply``.
    See that the name of the function is used for the name of its output,
    and an underscore-suffixed name for its function name.
    That's the default behavior if you don't specify either a name (of the function)
    for the `FuncNode`, or a `out`.
    The underscore is to distinguish from the name of the function itself.
    The function gets the underscore because this favors particular naming style.

    You can give it a custom name as well.

    >>> FuncNode(multiply, name='total_price', out='daily_expense')
    FuncNode(x,y -> total_price -> daily_expense)

    If you give an `out`, but not a `name` (for the function), the function's
    name will be taken:

    >>> FuncNode(multiply, out='daily_expense')
    FuncNode(x,y -> multiply -> daily_expense)

    If you give a `name`, but not a `out`, an underscore-prefixed version of
    the `name` will be taken:

    >>> FuncNode(multiply, name='total_price')
    FuncNode(x,y -> total_price -> _total_price)

    Note: In the context of networks if you want to reuse a same function
    (say, `multiply`) in multiple places
    you'll **need** to give it a custom name because the functions are identified by
    this name in the network.


    """

    # TODO: Make everything but func keyword-only (check for non-keyword usage before)
    # Using __init__ for now, but when 3.10, use field func with kw_only=True
    func: Callable
    name: str = field(default=None)
    bind: dict = field(default_factory=dict)
    out: str = field(default=None)
    func_label: str = field(default=None)  # TODO: Integrate more
    # write_output_into_scope: bool = True  # TODO: Do we really want to allow False?
    names_maker: Callable = underscore_func_node_names_maker
    node_validator: Callable = basic_node_validator

    # def __init__(
    #     self,
    #     func: Callable,
    #     *,
    #     name: str = None,
    #     bind: dict = None,
    #     out: str = None,
    #     func_label: str = None,  # TODO: Integrate more
    #     # write_output_into_scope: bool = True  # TODO: Do we really want to allow False?
    #     names_maker: Callable = underscore_func_node_names_maker,
    #     node_validator: Callable = basic_node_validator,
    # ):
    #     self.func = func
    #     self.name = name
    #     self.bind = bind
    #     self.out = out
    #     self.func_label = func_label
    #     # self.write_output_into_scope = write_output_into_scope
    #     self.names_maker = names_maker
    #     self.node_validator = node_validator
    #     self.__post_init__()

    def __post_init__(self):
        if self.bind is None:
            self.bind = dict()
        self.func = handle_variadics(self.func)
        _func_node_args_validation(func=self.func, name=self.name, out=self.out)
        self.name, self.out = self.names_maker(self.func, self.name, self.out)
        self.__name__ = self.name
        # self.__name__ = self.name
        # The wrapped function's signature will be useful
        # when interfacing with it and the scope.
        self.sig = Sig(self.func)

        # replace integer bind keys with their corresponding name
        self.bind = _bind_where_int_keys_repl_with_argname(self.bind, self.sig.names)
        # complete bind with the argnames of the signature
        _complete_dict_with_iterable_of_required_keys(self.bind, self.sig.names)
        _func_node_args_validation(bind=self.bind)

        self.extractor = partial(_mapped_extraction, to_extract=self.bind)

        if self.func_label is None:
            self.func_label = self.name

        self.node_validator(self)

    # TODO: BindInfo lists only three unique behaviors, but there are seven actual
    #  possible values for bind_info. All the rest are convenience aliases. Is this
    #  a good idea? The hesitation here comes from the fact that the values/keys
    #  language describes the bind data structure (dict), but the var_nodes/params
    #  language describes their contextual use. If had to choose, I'd chose the latter.
    def synopsis_string(self, bind_info: BindInfo = "values"):
        """

        :param bind_info: How to represent the bind in the synopsis string. Could be:
            - 'values', `var_nodes` or `varnodes`: the values of the bind (default).
            - 'keys' or 'params': the keys of the bind
            - 'hybrid': the keys of the bind, but with the values that are the same as
                the keys omitted.
        :return:

        >>> fn = FuncNode(
        ...     func=lambda y, c: None , name='h', bind={'y': 'b', 'c': 'c'}, out='d'
        ... )
        >>> fn.synopsis_string()
        'b,c -> h -> d'
        >>> fn.synopsis_string(bind_info='keys')
        'y,c -> h -> d'
        >>> fn.synopsis_string(bind_info='hybrid')
        'y=b,c -> h -> d'
        """
        if bind_info in {"values", "varnodes", "var_nodes"}:
            return f"{','.join(self.bind.values())} -> {self.name} " f"-> {self.out}"
        elif bind_info == "hybrid":

            def gen():
                for k, v in self.bind.items():
                    if k == v:
                        yield k
                    else:
                        yield f"{k}={v}"

            return f"{','.join(gen())} -> {self.name} " f"-> {self.out}"
        elif bind_info in {"keys", "params"}:
            return f"{','.join(self.bind.keys())} -> {self.name} " f"-> {self.out}"
        else:
            raise ValueError(f"Unknown bind_info: {bind_info}")

    def __repr__(self):
        return f'FuncNode({self.synopsis_string(bind_info="hybrid")})'

    def call_on_scope(self, scope: MutableMapping, write_output_into_scope=True):
        """Call the function using the given scope both to source arguments and write
        results.

        Note: This method is only meant to be used as a backend to __call__, not as
        an actual interface method. Additional control/constraints on read and writes
        can be implemented by providing a custom scope for that."""
        relevant_kwargs = dict(self.extractor(scope))
        args, kwargs = self.sig.mk_args_and_kwargs(relevant_kwargs)
        output = call_somewhat_forgivingly(
            self.func, args, kwargs, enforce_sig=self.sig
        )
        if write_output_into_scope:
            scope[self.out] = output
        return output

    def _hash_str(self):
        """Design idea.
        Attempt to construct a hash that reflects the actual identity we want.
        Need to transform to int. Only identifier chars alphanumerics and underscore
        and space are used, so could possibly encode as int (for __hash__ method)
        in a way that is reverse-decodable and with reasonable int size.
        """
        return self.synopsis_string(bind_info="hybrid")
        # return ';'.join(self.bind) + '::' + self.out

    # TODO: Find a better one. Need to have guidance on hash and eq methods dos-&-donts
    def __hash__(self):
        return hash(self._hash_str())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __call__(self, scope):
        """Deprecated: Don't use. Might be a normal function with a signature"""
        from warnings import warn

        raise DeprecationWarning(f"Deprecated. Use .call_on_scope(scope) instead.")
        # warn(f'Deprecated. Use .call_on_scope(scope) instead.', DeprecationWarning)
        # return self.call_on_scope(scope)

    def to_dict(self):
        """The inverse of from_dict: FuncNode.from_dict(fn.to_dict()) == fn"""
        return {x.name: getattr(self, x.name) for x in fields(self)}

    @classmethod
    def from_dict(cls, dictionary: dict):
        """The inverse of to_dict: Make a ``FuncNode`` from a dictionary of init args"""
        return cls(**dictionary)

    def ch_attrs(self, **new_attrs_values):
        """Returns a copy of the func node with some of its attributes changed

        >>> def plus(a, b):
        ...     return a + b
        ...
        >>> def minus(a, b):
        ...     return a - b
        ...
        >>> fn = FuncNode(func=plus, out='sum')
        >>> fn.func == plus
        True
        >>> fn.name == 'plus'
        True
        >>> new_fn = fn.ch_attrs(func=minus)
        >>> new_fn.func == minus
        True
        >>> new_fn.synopsis_string() == 'a,b -> plus -> sum'
        True
        >>>
        >>>
        >>> newer_fn = fn.ch_attrs(func=minus, name='sub', out='difference')
        >>> newer_fn.synopsis_string() == 'a,b -> sub -> difference'
        True
        """
        return ch_func_node_attrs(self, **new_attrs_values)

    @classmethod
    def has_as_instance(cls, obj):
        """Verify if ``obj`` is an instance of a FuncNode (or specific sub-class).

        The usefulness of this method is to not have to make a lambda with isinstance
        when filtering.

        >>> FuncNode.has_as_instance(FuncNode(lambda x: x))
        True
        >>> FuncNode.has_as_instance("I am not a FuncNode: I'm a string")
        False
        """
        return isinstance(obj, cls)

    def dot_lines(self, **kwargs):
        """Returns a list of lines that can be used to make a dot graph"""

        out = self.out

        func_id = self.name
        func_label = getattr(self, "func_label", func_id)
        if out == func_id:  # though forbidden in default FuncNode validation
            func_id = "_" + func_id

        # Get the Parameter objects for sig, with names changed to bind ones
        params = self.sig.ch_names(**self.bind).params

        yield from dot_lines_of_func_parameters(
            params, out=out, func_id=func_id, func_label=func_label, **kwargs
        )


# -------------------------------------------------------------------------------------
# viz stuff

from i2.signatures import Parameter, empty, Sig

# These are the defaults used in lined.
# TODO: Merge some of the functionalities around graph displays in lined and meshed
# TODO: Allow this to be overridden/edited by user, config2py style?
dflt_configs = dict(
    fnode_shape="box",
    vnode_shape="none",
    display_all_arguments=True,
    edge_kind="to_args_on_edge",
    input_node=True,
    output_node="output",
    func_display=True,
)


def dot_lines_of_func_parameters(
    parameters: Iterable[Parameter],
    out: str,
    func_id: str,
    *,
    func_label: str = None,
    vnode_shape: str = dflt_configs["vnode_shape"],
    fnode_shape: str = dflt_configs["fnode_shape"],
    func_display: bool = dflt_configs["func_display"],
) -> Iterable[str]:
    assert func_id != out, (
        f"Your func and output name shouldn't be the " f"same: {out=} {func_id=}"
    )
    yield f'{out} [label="{out}" shape="{vnode_shape}"]'
    for p in parameters:
        yield from param_to_dot_definition(p, shape=vnode_shape)

    if func_display:
        func_label = func_label or func_id
        yield f'{func_id} [label="{func_label}" shape="{fnode_shape}"]'
        yield f"{func_id} -> {out}"
        for p in parameters:
            yield f"{p.name} -> {func_id}"
    else:
        for p in parameters:
            yield f"{p.name} -> {out}"


def param_to_dot_definition(p: Parameter, shape=dflt_configs["vnode_shape"]):
    if p.default is not empty:
        name = p.name + "="
    elif p.kind == p.VAR_POSITIONAL:
        name = "*" + p.name
    elif p.kind == p.VAR_KEYWORD:
        name = "**" + p.name
    else:
        name = p.name
    yield f'{p.name} [label="{name}" shape="{shape}"]'


# -------------------------------------------------------------------------------------


@dataclass
class Mesh:
    func_nodes: Iterable[FuncNode]

    def synopsis_string(self, bind_info: BindInfo = "values"):
        return "\n".join(
            func_node.synopsis_string(bind_info) for func_node in self.func_nodes
        )


def validate_that_func_node_names_are_sane(func_nodes: Iterable[FuncNode]):
    """Assert that the names of func_nodes are sane.
    That is:

    * are valid dot (graphviz) names (we'll use str.isidentifier because lazy)
    * All the ``func.name`` and ``func.out`` are unique
    * more to come (TODO)...
    """
    func_nodes = list(func_nodes)
    node_names = [x.name for x in func_nodes]
    outs = [x.out for x in func_nodes]
    assert all(
        map(str.isidentifier, node_names)
    ), f"some node names weren't valid identifiers: {node_names}"
    assert all(
        map(str.isidentifier, outs)
    ), f"some return names weren't valid identifiers: {outs}"
    if len(set(node_names) | set(outs)) != 2 * len(func_nodes):
        c = Counter(node_names + outs)
        offending_names = [name for name, count in c.items() if count > 1]
        raise ValueError(
            f"Some of your node names and/or outs where used more than once. "
            f"They shouldn't. These are the names I find offensive: {offending_names}"
        )


def ensure_func_nodes(func_nodes):
    """Converts a list of objects to a list of FuncNodes."""
    # TODO: Take care of names (or track and take care if collision)
    if callable(func_nodes) and not isinstance(func_nodes, Iterable):
        # if input is a single function, make it a list containing that function
        single_func = func_nodes
        func_nodes = [single_func]
    for func_node in func_nodes:
        if is_func_node(func_node):
            yield func_node
        elif isinstance(func_node, Callable):
            yield FuncNode(func_node)
        else:
            raise TypeError(f"Can't convert this to a FuncNode: {func_node}")


_mk_func_nodes = ensure_func_nodes  # backwards compatibility


def _func_nodes_to_graph_dict(func_nodes):
    g = dict()

    for f in func_nodes:
        for src_name in f.bind.values():
            add_edge(g, src_name, f)
        add_edge(g, f, f.out)
    return g


def is_func_node(obj) -> bool:
    """
    >>> is_func_node(FuncNode(lambda x: x))
    True
    >>> is_func_node("I am not a FuncNode: I'm a string")
    False
    """
    # A weaker check than an isinstance(obj, FuncNode), which fails when we're
    # developing (therefore changing) FuncNode definition (without relaunching python
    # kernel). This is to be used instead, at least during development times
    # TODO: Replace with isinstance(obj, FuncNode) is this when development stabalizes
    #  See: https://github.com/i2mint/meshed/discussions/57
    # return isinstance(obj, FuncNode)
    cls = type(obj)
    if cls is not type:
        try:
            return any(getattr(x, "__name__", "") == "FuncNode" for x in cls.mro())
        except Exception:
            return isinstance(obj, FuncNode)
    else:
        return False


def is_not_func_node(obj) -> bool:
    """
    >>> is_not_func_node(FuncNode(lambda x: x))
    False
    >>> is_not_func_node("I am not a FuncNode: I'm a string")
    True
    """
    return not FuncNode.has_as_instance(obj)


def get_init_params_of_instance(obj):
    """Get names of instance object ``obj`` that are also parameters of the
    ``__init__`` of its class"""
    return {k: v for k, v in vars(obj).items() if k in Sig(type(obj)).names}


def ch_func_node_attrs(fn: FuncNode, **new_attrs_values):
    """Returns a copy of the func node with some of its attributes changed

    >>> def plus(a, b):
    ...     return a + b
    ...
    >>> def minus(a, b):
    ...     return a - b
    ...
    >>> fn = FuncNode(func=plus, out='sum')
    >>> fn.func == plus
    True
    >>> fn.name == 'plus'
    True
    >>> new_fn = ch_func_node_attrs(fn, func=minus)
    >>> new_fn.func == minus
    True
    >>> new_fn.synopsis_string() == 'a,b -> plus -> sum'
    True
    >>>
    >>>
    >>> newer_fn = ch_func_node_attrs(fn, func=minus, name='sub', out='difference')
    >>> newer_fn.synopsis_string() == 'a,b -> sub -> difference'
    True
    """
    init_params = get_init_params_of_instance(fn)
    if params_that_are_not_init_params := (new_attrs_values.keys() - init_params):
        raise ValueError(
            f"These are not params of {type(fn).__name__}: "
            f"{params_that_are_not_init_params}"
        )
    fn_kwargs = dict(init_params, **new_attrs_values)
    return FuncNode(**fn_kwargs)


def raise_signature_mismatch_error(fn, func):
    raise ValueError(
        "You can only change the func of a FuncNode with a another func if the "
        "signatures match.\n"
        f"\t{fn=}\n"
        f"\t{Sig(fn.func)=}\n"
        f"\t{Sig(func)=}\n"
    )


# from i2.signatures import keyed_comparator, SignatureComparator
# if compare_func is None:
#     compare_func = keyed_comparator(signature_comparator, key=Sig)


def _ch_func_node_func(fn: FuncNode, func: Callable):
    return ch_func_node_attrs(fn, func=func)


def ch_func_node_func(
    fn: FuncNode,
    func: Callable,
    *,
    func_comparator: CallableComparator = compare_signatures,
    ch_func_node=_ch_func_node_func,
    alternative=raise_signature_mismatch_error,
):
    if func_comparator(fn.func, func):
        return ch_func_node(fn, func=func)
    else:
        return alternative(fn, func)


def _new_bind(fnode, new_func):
    old_sig = Sig(fnode.func)
    new_sig = Sig(new_func)
    old_bind: dict = fnode.bind
    old_to_new_names_map = dict(zip(old_sig.names, new_sig.names))
    # TODO: assert some health stats on old_to_new_names_map
    new_bind = {old_to_new_names_map[k]: v for k, v in old_bind.items()}
    return new_bind


# TODO: Add more control (signature comparison, rebinding rules, renaming rules...)
# TODO: For example, can rebind to a function with different defaults, which are ignored.
#  Should we allow this? Should we allow to specify how to handle this?
# TODO: Should we include this in FuncNode as .ch_func(func)?
#  Possibly with an argument that specifies how to handle details, aligned with the
#  DAG.ch_funcs method. See ch_func_node_func.
def rebind_to_func(fnode: FuncNode, new_func: Callable):
    """Replaces ``fnode.func`` with ``new_func``, changing the ``.bind`` accordingly.

    >>> fn = FuncNode(lambda x, y: x + y, bind={'x': 'X', 'y': 'Y'})
    >>> fn.call_on_scope(dict(X=2, Y=3))
    5
    >>> new_fn = rebind_to_func(fn, lambda a, b, c=0: a * (b + c))
    >>> new_fn.call_on_scope(dict(X=2, Y=3))
    6
    >>> new_fn.call_on_scope(dict(X=2, Y=3, c=1))
    8
    """
    new_bind = _new_bind(fnode, new_func)
    return fnode.ch_attrs(func=new_func, bind=new_bind)


def insert_func_if_compatible(func_comparator: CallableComparator = compare_signatures):
    return partial(ch_func_node_func, func_comparator=func_comparator)


def _keys_and_values_are_strings_validation(d: dict):
    for k, v in d.items():
        if not isinstance(k, str):
            raise ValidationError(f"Should be a str: {k}")
        if not isinstance(v, str):
            raise ValidationError(f"Should be a str: {v}")


def _func_node_args_validation(
    *, func: Callable = None, name: str = None, bind: dict = None, out: str = None
):
    """Validates the four first arguments that are used to make a ``FuncNode``.
    Namely, if not ``None``,

    * ``func`` should be a callable

    * ``name`` and ``out`` should be ``str``

    * ``bind`` should be a ``Dict[str, str]``, ``Dict[int, str]`` or ``List[str]``

    * ``out`` should be a str

    """
    if func is not None and not isinstance(func, Callable):
        raise ValidationError(f"Should be callable: {func}")
    if name is not None and not isinstance(name, str):
        raise ValidationError(f"Should be a str: {name}")
    if bind is not None:
        if not isinstance(bind, dict):
            raise ValidationError(f"Should be a dict: {bind}")
        _keys_and_values_are_strings_validation(bind)
    if out is not None and not isinstance(out, str):
        raise ValidationError(f"Should be a str: {out}")


def _old_mapped_extraction(extract_from: dict, key_map: dict):
    """Deprecated: Old version of _mapped_extraction.

    for every (k, v) of key_map whose v is a key of extract_from, yields
    (v, extract_from[v])

    Meant to be curried into an extractor, and wrapped in dict.

    >>> extracted = _old_mapped_extraction(
    ...     {'a': 1, 'b': 2, 'c': 3}, # extract_from
    ...     {'A': 'a', 'C': 'c', 'D': 'd'}  # note that there's no 'd' in extract_from
    ... )
    >>> dict(extracted)
    {'a': 1, 'c': 3}

    """
    for k, v in key_map.items():
        if v in extract_from:
            yield v, extract_from[v]


def _mapped_extraction(src: dict, to_extract: dict):
    """for every (desired_name, src_name) of to_extract whose v is a key of source,
    yields (desired_name, source[src_name])

    It's purpose is to extract inputs from a src.
    The names used in the src may be different from those desired by the function,
    those to_extract specifies what to extract by a {desired_name: src_name, ...}
    map.

    _mapped_extraction_ is mant to be curried into an extractor.

    >>> extracted = _mapped_extraction(
    ...     src={'A': 1, 'B': 2, 'C': 3},
    ...     to_extract={'a': 'A', 'c': 'C', 'd': 'D'}  # note that there's no 'd' here
    ... )
    >>> dict(extracted)
    {'a': 1, 'c': 3}

    """
    for desired_name, src_name in to_extract.items():
        if src_name in src:
            yield desired_name, src[src_name]


def duplicates(elements: Union[Iterable, Sized]):
    c = Counter(elements)
    if len(c) != len(elements):
        return [name for name, count in c.items() if count > 1]
    else:
        return []


def _bind_where_int_keys_repl_with_argname(bind: dict, names: Sequence[str]) -> dict:
    """

    :param bind: A bind dict, as used in FuncNode
    :param names: A sequence of strings
    :return: A bind dict where integer keys were replaced with the corresponding
        name from names.

    >>> bind = {0: 'a', 1: 'b', 'c': 'x', 'd': 'y'}
    >>> names = 'e f g h'.split()
    >>> _bind_where_int_keys_repl_with_argname(bind, names)
    {'e': 'a', 'f': 'b', 'c': 'x', 'd': 'y'}
    """

    def transformed_items():
        for k, v in bind.items():
            if isinstance(k, int):
                argname = names[k]
                yield argname, v
            else:
                yield k, v

    return dict(transformed_items())


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


from typing import NewType, Dict, Tuple, Mapping

# TODO: Make a type where ``isinstance(s, Identifier) == s.isidentifier()``
Identifier = NewType("Identifier", str)  # + should satisfy str.isidentifier
Bind = Union[
    str,  # Identifier or ' '.join(Iterable[Identifier])
    Dict[Identifier, Identifier],
    Sequence[Union[Identifier, Tuple[Identifier, Identifier]]],
]

IdentifierMapping = Dict[Identifier, Identifier]


def identifier_mapping(x: Bind) -> IdentifierMapping:
    """Get an ``IdentifierMapping`` dict from a more loosely defined ``Bind``.

    You can get an identifier mapping (that is, an explicit for for a ``bind`` argument)
    from...

    ... a single space-separated string

    >>> identifier_mapping('x a_b yz')  #
    {'x': 'x', 'a_b': 'a_b', 'yz': 'yz'}

    ... an iterable of strings or pairs of strings

    >>> identifier_mapping(['foo', ('bar', 'mitzvah')])
    {'foo': 'foo', 'bar': 'mitzvah'}

    ... a dict will be considered to be the mapping itself

    >>> identifier_mapping({'x': 'y', 'a': 'b'})
    {'x': 'y', 'a': 'b'}
    """
    if isinstance(x, str):
        x = x.split()
    if not isinstance(x, Mapping):

        def gen():
            for item in x:
                if isinstance(item, str):
                    yield item, item
                else:
                    yield item

        return dict(gen())
    else:
        return dict(**x)


FuncNodeAble = Union[FuncNode, Callable]


def func_node_transformer(
    fn: FuncNode,
    kwargs_transformers=(),
):
    """Get a modified ``FuncNode`` from an iterable of ``kwargs_trans`` modifiers."""
    func_node_kwargs = fn.to_dict()
    if callable(kwargs_transformers):
        kwargs_transformers = [kwargs_transformers]
    for trans in kwargs_transformers:
        if (new_kwargs := trans(func_node_kwargs)) is not None:
            func_node_kwargs = new_kwargs
    return FuncNode.from_dict(func_node_kwargs)


def func_nodes_to_code(
    func_nodes: Iterable[FuncNode],
    func_name: str = "generated_pipeline",
    *,
    favor_positional: bool = True,
) -> str:
    """Convert an iterable of FuncNodes back to executable Python code.

    This is the inverse operation of code_to_fnodes - it takes FuncNodes and generates
    Python code that would create equivalent FuncNodes when parsed.
    When favor_positional is True, any keyword argument with key equal to its value
    is moved to the positional arguments list:

        func(a=a, b=b, c=z, d=d)  ->  func(a, b, c=z, d=d)

    :param func_nodes: Iterable of FuncNode instances to convert to code
    :param func_name: Name for the generated function
    :param favor_positional: When True, transforms kwargs of the form key=key into positional args.
    :return: String containing Python code


    """

    def lines():
        yield f"def {func_name}():"
        for func_node in func_nodes:
            line = _func_node_to_assignment_line(
                func_node, favor_positional=favor_positional
            )
            yield f"    {line}"

    return "\n".join(lines())


def _func_node_to_assignment_line(func_node: FuncNode, favor_positional=True) -> str:
    """Convert a single FuncNode to a Python assignment line.

    :param func_node: The FuncNode to convert
    :param favor_positional: When True, transforms kwargs of the form key=key into positional args
    :return: String like "output = func_name(arg1, arg2=value)"
    """
    # Get the function name - use func_label if available, otherwise name
    func_name = getattr(func_node, "func_label", None) or func_node.name

    # Handle special cases for generated functions
    if func_name.endswith("__0") or func_name.endswith("__1"):
        # This is likely an itemgetter from tuple unpacking
        if hasattr(func_node.func, "keywords") and "keys" in func_node.func.keywords:
            keys = func_node.func.keywords["keys"]
            if len(keys) == 1:
                # Single item extraction
                source_var = list(func_node.bind.values())[0]
                return f"{func_node.out} = {source_var}[{keys[0]}]"

    # Build argument list
    args = []
    kwargs = []

    # Process bind dictionary to separate positional and keyword arguments
    for param, source in func_node.bind.items():
        if isinstance(param, int):
            # Positional argument
            args.append((param, source))
        else:
            # Keyword argument - check if favor_positional applies
            if favor_positional and param == source:
                # Convert to positional argument by using the parameter order from signature
                try:
                    param_index = list(func_node.sig.names).index(param)
                    args.append((param_index, source))
                except (ValueError, AttributeError):
                    # If we can't determine order, keep as keyword
                    kwargs.append((param, source))
            else:
                kwargs.append((param, source))

    # Sort positional arguments by their index
    args.sort(key=lambda x: x[0])

    # Build the argument string
    arg_parts = []

    # Add positional arguments
    for _, source in args:
        arg_parts.append(str(source))

    # Add keyword arguments
    for param, source in kwargs:
        arg_parts.append(f"{param}={source}")

    arg_string = ", ".join(arg_parts)

    # Handle tuple unpacking in output
    if "__" in func_node.out and not func_node.out.endswith(("__0", "__1")):
        # This might be a tuple output that was created from tuple unpacking
        output_parts = func_node.out.split("__")
        if len(output_parts) > 1:
            output = ", ".join(output_parts)
            return f"{output} = {func_name}({arg_string})"

    return f"{func_node.out} = {func_name}({arg_string})"
