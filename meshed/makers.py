r"""Makers

This module contains tools to make meshed objects in different ways.

Let's start with an example where we have some code representing a user story:

>>> def user_story():
...     wfs = call(src_to_wf, data_src)
...     chks_iter = map(chunker, wfs)
...     chks = chain(chks_iter)
...     fvs = map(featurizer, chks)
...     model_outputs = map(model, fvs)

If the code is compliant (has only function calls and assignments of their result),
we can extract ``FuncNode`` factories from these lines (uses AST behind the scenes).

>>> from meshed.makers import src_to_func_node_factory
>>> fnodes_factories = list(src_to_func_node_factory(user_story))

Each factory is a curried version of ``FuncNode``, set up to be able to make a ``DAG``
equivalent to the user story, once we provide the necessary functions (``call``,
``map``, and ``chain``).

>>> from functools import partial
>>> assert all(
... isinstance(x, partial) and issubclass(x.func, FuncNode) for x in fnodes_factories
... )

See that the ``FuncNode`` factories are all set up with
``name`` (id),
``out`` (output variable name),
``bind`` (names of the variables where the function will source it's arguments), and
``func_label`` (which can be used when displaying the DAG, or as a key to the function
to use).

>>> assert [x.keywords for x in fnodes_factories] == [
...  {'name': 'call',
...   'out': 'wfs',
...   'bind': {0: 'src_to_wf', 1: 'data_src'},
...   'func_label': 'call'},
...  {'name': 'map',
...   'out': 'chks_iter',
...   'bind': {0: 'chunker', 1: 'wfs'},
...   'func_label': 'map'},
...  {'name': 'chain',
...   'out': 'chks',
...   'bind': {0: 'chks_iter'},
...   'func_label': 'chain'},
...  {'name': 'map_04',
...   'out': 'fvs',
...   'bind': {0: 'featurizer', 1: 'chks'},
...   'func_label': 'map'},
...  {'name': 'map_05',
...   'out': 'model_outputs',
...   'bind': {0: 'model', 1: 'fvs'},
...   'func_label': 'map'}
... ]

What can we do with that?

Well, provide the functions, so the DAG can actually compute.

You can do it yourself, or get a little help with ``mk_fnodes_from_fn_factories``.

>>> from meshed.dag import DAG
>>> from meshed.makers import mk_fnodes_from_fn_factories
>>> fnodes = list(mk_fnodes_from_fn_factories(fnodes_factories))
>>> dag = DAG(fnodes)
>>> print(dag.synopsis_string())
src_to_wf,data_src -> call -> wfs
chunker,wfs -> map -> chks_iter
chks_iter -> chain -> chks
featurizer,chks -> map_04 -> fvs
model,fvs -> map_05 -> model_outputs

Wait! But we didn't actually provide the functions we wanted to use!
What happened?!?
What happened is that ``mk_fnodes_from_fn_factories`` just made some for us.
It used the convenient ``meshed.util.mk_place_holder_func`` which makes a function
(that happens to actually compute something and be picklable).

>>> from inspect import signature
>>> str(signature(dag))
'(src_to_wf, data_src, chunker, featurizer, model)'

We can actually call the ``dag`` and get something meaningful:

>>> dag(1, 2, 3, 4, 5)
'map(model=5, fvs=map(featurizer=4, chks=chain(chks_iter=map(chunker=3, wfs=call(src_to_wf=1, data_src=2)))))'

If you don't want ``mk_fnodes_from_fn_factories`` to do that (because you are in
prod and need to make sure as much as possible is explicitly as expected, you can
simply use a different ``factory_to_func`` argument. The default one is:

>>> from meshed.makers import dlft_factory_to_func

which you can also reuse to make your own.
See below how we provide a ``name_to_func_map`` to specify how ``func_label``s should
map to actual functions, and set ``use_place_holder_fallback=False`` to make
sure that we don't ever fallback on a placeholder function as we did above.

>>> def _call(x, y):
...     # would use operator.methodcaller('__call__') but doesn't have a __name__
...     return x + y
>>> def _map(x, y):
...     return [x, y]
>>> def _chain(iterable):
...     return sum(iterable)
>>>
>>> factory_to_func = partial(
...     dlft_factory_to_func,
...     name_to_func_map={'map': _map, 'chain': _chain, 'call': _call},
...     use_place_holder_fallback=False
... )
>>>
>>> fnodes = list(mk_fnodes_from_fn_factories(fnodes_factories, factory_to_func))
>>> dag = DAG(fnodes)

On the surface, we get the same dag as we had before -- at least from the point of view
of the dag signature, names, and relationships between these names:

>>> print(dag.synopsis_string())
src_to_wf,data_src -> call -> wfs
chunker,wfs -> map -> chks_iter
chks_iter -> chain -> chks
featurizer,chks -> map_04 -> fvs
model,fvs -> map_05 -> model_outputs
>>> str(signature(dag))
'(src_to_wf, data_src, chunker, featurizer, model)'

But see below that the dag is now using the functions we specified:

>>> # dag(src_to_wf=1, data_src=2, chunker=3, featurizer=4, model=5)
>>> # will trigger this:
>>> # src_to_wf=1, data_src=2 -> call -> wfs == 1 + 2 == 3
>>> # chunker=3 , wfs=3 -> map -> chks_iter == [3, 3]
>>> # chks_iter=6 -> chain -> chks == 3 + 3 == 6
>>> # featurizer=4, chks=6 -> map_04 -> fvs == [4, 6]
>>> # model=5, fvs=[4, 6] -> map_05 -> model_outputs == [5, [4, 6]]
>>> dag(1, 2, 3, 4, 5)
[5, [4, 6]]

"""


import ast
import inspect
from operator import itemgetter, attrgetter
from typing import (
    Tuple,
    Optional,
    Iterator,
    Iterable,
    TypeVar,
    Callable,
    Dict,
    Mapping,
    Union,
)
from contextlib import suppress
from functools import partial


from i2 import Sig, name_of_obj, partialx

from meshed.dag import DAG
from meshed.base import FuncNode
from meshed.util import mk_place_holder_func


T = TypeVar('T')

# Some restrictions exist and need to be clarified or removed (i.e. more cases handled)
# For example,
# * tuple assignment not handled (x, y = func(...))
# * can't reuse a variable (would lead to same node)
# * x = y (or x, y = tup) not handled (but could easily by binding)
# We don't need these cases to be handled, only x = func(...) forms lead to Turing (I
# think...):
# Further other cases are not handled, but we don't want to handle ALL of python
# -- just a sufficiently expressive subset.


def attr_dict(obj):
    return {a: getattr(obj, a) for a in dir(obj) if not a.startswith('_')}


def is_from_ast_module(o):
    return getattr(type(o), '__module__', '').startswith('_ast')


def _ast_info_str(x):
    return f'lineno={x.lineno}'


def _itemgetter(items, item_keys=()):
    return tuple(items[i] for i in item_keys)


def signed_itemgetter(*item_keys):
    """Like ``operator.itemgetter``, except has a signature, which we needed"""
    return partialx(_itemgetter, item_keys=item_keys, _rm_partialize=True)


# Note: generalize? glom?
def parse_assignment(body: ast.Assign) -> Tuple:
    # TODO: Make this validation better (at least more help in raised error)
    # TODO: extract validating code out as validation functions?
    info = _ast_info_str(body)
    if not isinstance(body, ast.Assign):
        raise ValueError(f"All commands should be assignments, this one wasn't: {info}")

    target = body.targets
    assert len(target) == 1, f'Only one target allowed: {info}'
    target = target[0]
    assert isinstance(
        target, (ast.Name, ast.Tuple)
    ), f'Should be a ast.Name or ast.Tuple: {info}'

    value = body.value
    assert isinstance(value, ast.Call), f'Only one target allowed: {info}'

    return target, value


# TODO: Evolve this: Perhaps it can be used to centralize this concern:
def _extract_value_from_ast_element(ast_element):
    if isinstance(ast_element, ast.Name):
        return ast_element.id
    else:
        return ast_element.value


def parsed_to_node_kwargs(target_value) -> Iterator[dict]:
    """Extract FuncNode kwargs (name, out, and bind) from ast (target,value) pairs

    :param target_value: A (target, value) pair
    :return: A ``{name:..., out:..., bind:...}`` dict (meant to be used to curry FuncNode

    Where can you make make target_values? With the ``parse_assignment_steps`` function.

    >>> from meshed.makers import parse_assignment_steps
    >>> def foo():
    ...     x = func1(a, b=2)
    ...     y = func2(x, func1, c=3, d=x)
    >>> for target_value in parse_assignment_steps(foo):
    ...     for d in parsed_to_node_kwargs(target_value):
    ...         print(d)
    {'name': 'func1', 'out': 'x', 'bind': {0: 'a', 'b': 2}}
    {'name': 'func2', 'out': 'y', 'bind': {0: 'x', 1: 'func1', 'c': 3, 'd': 'x'}}

    """
    # Note: ast.Tuple has names in 'elts' attribute,
    # and could be handled, but would need to lead to multiple nodes
    target, value = target_value
    args = value.args
    bind_from_args = {i: k.id for i, k in enumerate(args)}
    kwargs = {x.arg: _extract_value_from_ast_element(x.value) for x in value.keywords}
    if isinstance(target, ast.Name):
        yield dict(
            name=value.func.id, out=target.id, bind=dict(bind_from_args, **kwargs)
        )
    elif isinstance(target, ast.Tuple):
        assign_to_names = tuple(map(attrgetter('id'), target.elts))
        # yield the function call information, assigning to a single variable
        # TODO: Long. Better way? (careful: need global uniqueness!)
        func_output_name = '__'.join(assign_to_names)
        yield dict(
            name=value.func.id,
            out=func_output_name,
            bind=dict(bind_from_args, **kwargs),
        )
        # then, yield instructions to extract variable into several
        for i, assign_to_name in enumerate(assign_to_names):
            yield dict(
                func=signed_itemgetter(i),
                name=f'{assign_to_name}__{i}',
                out=assign_to_name,
                bind={0: func_output_name},
                func_label=f'[{i}]',
            )
        # raise ValueError(f"You're here: {target=}")
    else:
        raise TypeError(f'Should be a ast.Name or ast.Tuple. Was: {target}')


FuncNodeFactory = Callable[[Callable], FuncNode]


def node_kwargs_to_func_node_factory(node_kwargs) -> FuncNodeFactory:
    return partial(FuncNode, **node_kwargs)


class AssignNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.store = []

    def visit_Assign(self, node):
        self.store.append(parse_assignment(node))
        return node


def retrieve_assignments(src):
    if callable(src):
        src = inspect.getsource(src)
    nodes = ast.parse(src)
    visitor = AssignNodeVisitor()
    visitor.visit(nodes)

    return visitor.store


def parse_assignment_steps(src):
    """
    Parse source code and generate tuples of information about it.

    :param src: The source string or a python object whose code string can be extracted.
    :return: And generator of "target_values"

    >>> from meshed.makers import parse_assignment_steps
    >>> def foo():
    ...     x = func1(a, b=2)
    ...     y = func2(x, c=3)
    >>> target_values = list(parse_assignment_steps(foo))

    Let's look at the first target_value to see what it contains:

    >>> name, call = target_values[0]  # a 2-tuple
    >>> assert isinstance(name, ast.Name)  # the first element is a ast Name object
    >>> sorted(vars(name))
    ['col_offset', 'ctx', 'end_col_offset', 'end_lineno', 'id', 'lineno']
    >>> name.id
    'x'
    >>> assert isinstance(call, ast.Call)  # the first element is a ast Call object
    >>> sorted(vars(call))
    ['args', 'col_offset', 'end_col_offset', 'end_lineno', 'func', 'keywords', 'lineno']
    >>> call.args[0].id
    'a'
    >>> call.keywords[0].arg
    'b'
    >>> call.keywords[0].value.value
    2

    Basically, these ast objects contain all we need to know about the (parsed) source.

    """
    if callable(src):
        src = inspect.getsource(src)
    root = ast.parse(src)
    assert len(root.body) == 1
    func_body = root.body[0]
    # TODO: work with func_body.args to get info on interface (name, args, kwargs,
    #  return etc.)
    #     return func_body
    for body in func_body.body:
        yield parse_assignment(body)


iterize = lambda func: partial(map, func)


FuncNodeFactory = Callable[..., FuncNode]
FactoryToFunc = Callable[[FuncNodeFactory], Callable]


def src_to_func_node_factory(
    src, names_used_so_far=None
) -> Iterator[Union[FuncNode, FuncNodeFactory]]:
    """A generator of FuncNode factories from a src (string or object)."""
    names_used_so_far = names_used_so_far or set()
    for i, target_value in enumerate(parse_assignment_steps(src), 1):
        for node_kwargs in parsed_to_node_kwargs(target_value):
            node_kwargs['func_label'] = node_kwargs['name']
            if node_kwargs['name'] in names_used_so_far:
                # need to keep names uniques, so add a prefix to (hope) to get uniqueness
                node_kwargs['name'] += f'_{i:02.0f}'
            names_used_so_far.add(node_kwargs['name'])
            yield node_kwargs_to_func_node_factory(node_kwargs)


dlft_factory_to_func: FactoryToFunc


# TODO: A bit strange to ask a factory for information to get a func that it needs
#  to make itself. Do we gain much over simply saying "factory, make yourself"?
def dlft_factory_to_func(
    factory: partial,
    name_to_func_map: Optional[Dict[str, Callable]] = None,
    use_place_holder_fallback=True,
):
    """Get a function for the given factory, using"""
    # TODO: Add extra validation (like n_args of return func against bind)
    name_to_func_map = name_to_func_map or dict()

    factory_kwargs = factory.keywords
    name = (
        factory_kwargs['func_label'] or factory_kwargs['name'] or factory_kwargs['out']
    )
    if name in name_to_func_map:
        return name_to_func_map[name]
    elif use_place_holder_fallback:
        arg_names = [
            k if isinstance(k, str) else v for k, v in factory_kwargs['bind'].items()
        ]
        return mk_place_holder_func(arg_names, name=name)
    else:
        raise KeyError(f'name not found in name_to_func_map: {name}')


def mk_fnodes_from_fn_factories(
    fnodes_factories: Iterable[FuncNodeFactory],
    factory_to_func: FactoryToFunc = dlft_factory_to_func,
) -> Iterator[FuncNode]:
    """Make func nodes from func node factories and a specification of how to make the
    nodes from these.

    :param fnodes_factories: An iterable of FuncNodeFactory
    :param factory_to_func: A function that will give you a function given a
        FuncNodeFactory input (where it will draw the information it needs to know
        what kind of function to make).
    :return:
    """
    # TODO: Might be a cleaner design for this...
    for fnode_factory in fnodes_factories:
        sig = Sig(fnode_factory)
        if sig.n_required == 1 and sig.names[0] == 'func':
            # first making sure the fnode_factory is exactly as expected for this case,
            # get a function for this fnode_factory, then use it to make the fnode
            func = factory_to_func(fnode_factory)
            yield fnode_factory(func)
        elif sig.n_required == 0:
            # if fnode_factory has no (required) arguments, just call the factory:
            yield fnode_factory()
        else:
            # if couldn't figure it out from the last two cases, freak out!
            raise ValueError(
                f"The fnode_factory didn't have the expected format, so I'm freaking "
                f"out. It's supposed to be a no-arguments-required-callable or a "
                f'functools.partial that needs only a func to make the func node. '
                f'This is the offending fnode_factory: {fnode_factory}'
            )


def _code_to_fnodes(src, func_src=dlft_factory_to_func):
    if isinstance(func_src, Mapping):
        name_to_func_map = func_src
        func_src = partial(
            dlft_factory_to_func,
            name_to_func_map=name_to_func_map,
            use_place_holder_fallback=False,
        )
    assert isinstance(func_src, Callable), f'func_src should be callable, or a mapping'
    fnodes_factories = list(src_to_func_node_factory(src))
    return mk_fnodes_from_fn_factories(fnodes_factories, func_src)


def code_to_dag(src, func_src=dlft_factory_to_func, name=None):
    """Get a ``meshed.DAG`` from src code"""
    if name is None:
        if isinstance(src, str):
            name = 'dag_made_from_code_parsing'
        else:
            name = name_of_obj(src)
    fnodes = _code_to_fnodes(src, func_src)
    return DAG(fnodes, name=name)


def code_to_digraph(src):
    return code_to_dag(src).dot_digraph()


simple_code_to_digraph = code_to_digraph  # back-compatability alias


def _old_simple_code_to_digraph_with_regex(code):
    """Make a graphviz.Digraph object based on code (string or function's body)"""
    import re
    from inspect import getsource
    from graphviz import Digraph

    def get_code_str(code) -> str:
        if not isinstance(code, str):
            return getsource(code)
        return code

    empty_spaces = re.compile('^\s*$')
    simple_assignment_p = re.compile(
        '(?P<output_vars>[^=]+)' '\s*=\s*' '(?P<func>\w+)' '\((?P<input_vars>.*)\)'
    )

    def get_lines(code_str):
        for line in code_str.split('\n'):
            if not empty_spaces.match(line):
                yield line.strip()

    def groupdict_parser(s, pattern):
        pattern = re.compile(pattern)
        m = pattern.search(s)
        if m:
            return m.groupdict()
        return None

    def parsed_lines(
        code_str, line_parser=partial(groupdict_parser, pattern=simple_assignment_p)
    ):
        yield from filter(None, map(line_parser, get_lines(code_str)))

    def parsed_lines_to_dot(parsed_lines):
        for d in parsed_lines:
            yield f"{d['func']} [shape=box]"
            yield f"{d['input_vars']} -> {d['func']} -> {d['output_vars']}"

    code_str = get_code_str(code)
    dot_lines = parsed_lines_to_dot(parsed_lines(code_str))
    return Digraph(body=dot_lines)


# TODO: Replace using builtin random
with suppress(ModuleNotFoundError, ImportError):
    from numpy.random import randint, choice

    def random_graph(n_nodes=7):
        """Get a random graph"""
        nodes = range(n_nodes)

        def gen():
            for src in nodes:
                n_dst = randint(0, n_nodes - 1)
                dst = choice(n_nodes, n_dst, replace=False)
                yield src, list(dst)

        return dict(gen())
