"""
Base functionality of meshed
"""
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, MutableMapping, Iterable, Union, Sized, Sequence

from i2 import Sig, call_somewhat_forgivingly
from meshed.util import ValidationError, NameValidationError, mk_func_name
from meshed.itools import add_edge


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
    if name is not None and out is not None:
        return name, out

    try:
        name_of_func = mk_func_name(func)
    except NameValidationError as err:
        err_msg = err.args[0]
        err_msg += (
            f'\nSuggestion: You might want to specify a name explicitly in '
            f'FuncNode(func, name=name) instead of just giving me the func as is.'
        )
        raise NameValidationError(err_msg)
    if name is None and out is None:
        return name_of_func + '_', name_of_func
    elif out is None:
        return name, '_' + name
    elif name is None:
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
        names_that_are_not_strings = ', '.join(map(str, names_that_are_not_strings))
        raise ValidationError(f'Should be strings: {names_that_are_not_strings}')

    # Make sure there's no name duplicates
    _duplicates = duplicates(names)
    if _duplicates:
        raise ValidationError(f'{func_node} has duplicate names: {_duplicates}')

    # Make sure all names are identifiers
    _non_identifiers = list(filter(lambda name: not name.isidentifier(), names))
    # print(_non_identifiers, names)
    if _non_identifiers:
        raise ValidationError(f'{func_node} non-identifier names: {_non_identifiers}')

    # Making sure all src_name keys are in the function's signature
    bind_names_not_in_sig_names = func_node.bind.keys() - func_node.sig.names
    assert not bind_names_not_in_sig_names, (
        f"some bind keys weren't found as function argnames: "
        f"{', '.join(bind_names_not_in_sig_names)}"
    )


# TODO: Think of the hash more carefully.
@dataclass
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
    FuncNode(item_price,num_of_items -> multiply_ -> multiply)

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
    >>> func_node(scope)  # see that it returns 7.0
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

    func: Callable
    name: str = field(default=None)
    bind: dict = field(default_factory=dict)
    out: str = field(default=None)
    func_label: str = field(default=None)  # TODO: Integrate more
    write_output_into_scope: bool = True  # TODO: Do we really want to allow False?
    names_maker: Callable = underscore_func_node_names_maker
    node_validator: Callable = basic_node_validator

    def __post_init__(self):
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

    def synopsis_string(self):
        return f"{','.join(self.bind.values())} -> {self.name} " f'-> {self.out}'

    def __repr__(self):
        return f'FuncNode({self.synopsis_string()})'

    def call_on_scope(self, scope: MutableMapping):
        """Call the function using the given scope both to source arguments and write
        results.

        Note: This method is only meant to be used as a backend to __call__, not as
        an actual interface method. Additional control/constraints on read and writes
        can be implemented by providing a custom scope for that."""
        relevant_kwargs = dict(self.extractor(scope))
        args, kwargs = self.sig.args_and_kwargs_from_kwargs(relevant_kwargs)
        output = call_somewhat_forgivingly(
            self.func, args, kwargs, enforce_sig=self.sig
        )
        if self.write_output_into_scope:
            scope[self.out] = output
        return output

    def _hash_str(self):
        """Design ideo.
        Attempt to construct a hash that reflects the actual identity we want.
        Need to transform to int. Only identifier chars alphanumerics and underscore
        and space are used, so could possibly encode as int (for __hash__ method)
        in a way that is reverse-decodable and with reasonable int size.
        """
        return ';'.join(self.bind) + '::' + self.out

    # TODO: Find a better one
    def __hash__(self):
        return hash(self._hash_str())

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __gt__(self, other):
        return hash(self) > hash(other)

    def __call__(self, scope):
        """Deprecated: Don't use. Might be a normal function with a signature"""
        return self.call_on_scope(scope)

    # See https://github.com/i2mint/meshed/issues/21 (not 12!)
    # def __eq__(self, other):
    #     return hash(self) == hash(other)

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
            f'Some of your node names and/or outs where used more than once. '
            f"They shouldn't. These are the names I find offensive: {offending_names}"
        )


def _mk_func_nodes(func_nodes):
    # TODO: Take care of names (or track and take care if collision)
    for func_node in func_nodes:
        if is_func_node(func_node):
            yield func_node
        elif isinstance(func_node, Callable):
            yield FuncNode(func_node)
        else:
            raise TypeError(f"Can't convert this to a FuncNode: {func_node}")


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
    # TODO: Replace with isinstance(obj, FuncNode) is this when development
    #  stabalizes
    # return isinstance(obj, FuncNode)
    cls = type(obj)
    if cls is not type:
        return any(getattr(x, '__name__', '') == 'FuncNode' for x in cls.mro())
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


def ch_func_node_attrs(fn, **new_attrs_values):
    """Returns a copy of the func node with some of it's attributes changed

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
            f'These are not params of {type(fn).__name__}: '
            f'{params_that_are_not_init_params}'
        )
    fn_kwargs = dict(init_params, **new_attrs_values)
    return FuncNode(**fn_kwargs)


def _keys_and_values_are_strings_validation(d: dict):
    for k, v in d.items():
        if not isinstance(k, str):
            raise ValidationError(f'Should be a str: {k}')
        if not isinstance(v, str):
            raise ValidationError(f'Should be a str: {v}')


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
        raise ValidationError(f'Should be callable: {func}')
    if name is not None and not isinstance(name, str):
        raise ValidationError(f'Should be a str: {name}')
    if bind is not None:
        if not isinstance(bind, dict):
            raise ValidationError(f'Should be a dict: {bind}')
        _keys_and_values_are_strings_validation(bind)
    if out is not None and not isinstance(out, str):
        raise ValidationError(f'Should be a str: {out}')


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
Identifier = NewType('Identifier', str)  # + should satisfy str.isidentifier
Bind = NewType(
    'Bind',
    Union[
        str,  # Identifier or ' '.join(Iterable[Identifier])
        Dict[Identifier, Identifier],
        Sequence[Union[Identifier, Tuple[Identifier, Identifier]]],
    ],
)
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
