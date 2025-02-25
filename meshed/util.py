"""util functions"""

import re
from functools import partial, wraps
from inspect import Parameter, getmodule
from types import ModuleType
from typing import (
    Callable,
    Any,
    Union,
    Iterator,
    Optional,
    Iterable,
    Mapping,
    TypeVar,
    Tuple,
    List,
)
from importlib import import_module
from operator import itemgetter

from i2 import Sig, name_of_obj, LiteralVal, FuncFanout, Pipe

T = TypeVar("T")


def objects_defined_in_module(
    module: Union[str, ModuleType],
    *,
    name_filt: Optional[Callable] = None,
    obj_filt: Optional[Callable] = None,
):
    """
    Get a dictionary of objects defined in a Python module, optionally filtered by their names and values.

    Parameters
    ----------
    module: Union[str, ModuleType]
        The module to look up. Can either be
        - the module object itself,
        - a string specifying the module's fully qualified name (e.g., 'os.path'), or
        - a .py filepath to the module

    name_filt: Optional[Callable], default=None
        An optional function used to filter the names of objects in the module.
        This function should take a single argument (the object name as a string)
        and return a boolean. Only objects whose names pass the filter (i.e.,
        for which the function returns True) are included.
        If None, no name filtering is applied.

    obj_filt: Optional[Callable], default=None
        An optional function used to filter the objects in the module. This function should take a
        single argument (the object itself) and return a boolean. Only objects that pass the filter
        (i.e., for which the function returns True) are included.
        If None, no object filtering is applied.

    Returns
    -------
    dict
        A dictionary where keys are names of objects defined in the module (filtered by name_filt and obj_filt)
        and values are the corresponding objects.

    Examples
    --------
    >>> import os
    >>> all_os_objects = objects_defined_in_module(os)
    >>> 'removedirs' in all_os_objects
    True
    >>> all_os_objects['removedirs'] == os.removedirs
    True

    See that you can specify the module via a string too, and filter to get only
    callables that don't start with an underscore:

    >>> this_modules_funcs = objects_defined_in_module(
    ...     'meshed.util',
    ...     name_filt=lambda name: not name.startswith('_'),
    ...     obj_filt=callable,
    ... )
    >>> callable(this_modules_funcs['objects_defined_in_module'])
    True

    """
    if isinstance(module, str):
        if module.endswith(".py") and os.path.isfile(module):
            module_filepath = module
            with filepath_to_module(module_filepath) as module:
                return objects_defined_in_module(
                    module, name_filt=name_filt, obj_filt=obj_filt
                )
        else:
            module = import_module(module)

    # At this point we have a module object (ModuleType)
    name_filt = name_filt or (lambda x: True)
    obj_filt = obj_filt or (lambda x: True)
    module_objs = vars(module)
    # Note we only filter for names here, not objects, because we want to keep the
    # object filtering for after we've gotten the module objects
    name_and_module = {
        name: getmodule(obj)
        for name, obj in module_objs.items()
        if name_filt(name) and obj is not None
    }
    obj_names = [
        obj_name
        for obj_name, obj_module in name_and_module.items()
        if obj_module is not None and obj_module.__name__ == module.__name__
    ]
    return {k: module_objs[k] for k in obj_names if obj_filt(module_objs[k])}


import importlib.util
import sys
import os
from contextlib import contextmanager


@contextmanager
def filepath_to_module(file_path: str):
    """
    A context manager to import a Python file as a module.

    :param file_path: The file path of the Python file to import.
    :yield: The module object.
    """
    file_path = os.path.abspath(file_path)
    dir_path = os.path.dirname(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        raise ImportError(f"Module {file_path} could not be imported.")

    # Add the directory of the file to sys.path
    sys.path.insert(0, dir_path)

    try:
        # Create a module from the spec and execute its code
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module

    finally:
        # Clean up: remove the added directory from sys.path
        sys.path.remove(dir_path)


def provides(*var_names: str) -> Callable[[Callable], Callable]:
    """Decorator to assign ``var_names`` to a ``_provides`` attribute of function.

    This is meant to be used to indicate to a mesh what var nodes a function can source
    values for.

    >>> @provides('a', 'b')
    ... def f(x):
    ...     return x + 1
    >>> f._provides
    ('a', 'b')

    If no ``var_names`` are given, then the function name is used as the var name:

    >>> @provides()
    ... def g(x):
    ...     return x + 1
    >>> g._provides
    ('g',)

    If ``var_names`` contains ``'_'``, then the function name is used as the var name
    for that position:

    >>> @provides('b', '_')
    ... def h(x):
    ...     return x + 1
    >>> h._provides
    ('b', 'h')

    """

    def add_provides_attribute(func):
        if not var_names:
            var_names_ = (name_of_obj(func),)
        else:
            var_names_ = tuple(
                [x if x != "_" else name_of_obj(func) for x in var_names]
            )
        func._provides = var_names_
        return func

    return add_provides_attribute


def if_then_else(if_func, then_func, else_func, *args, **kwargs):
    """
    Tool to "functionalize" the if-then-else logic.

    >>> from functools import partial
    >>> f = partial(if_then_else, str.isnumeric, int, str)
    >>> f('a string')
    'a string'
    >>> f('42')
    42

    """
    if if_func(*args, **kwargs):
        return then_func(*args, **kwargs)
    else:
        return else_func(*args, **kwargs)


# TODO: Revise FuncFanout so it makes a generator of values, or items, instead of a dict
def funcs_conjunction(*funcs):
    """
    Makes a conjunction of functions. That is, ``func1(x) and func2(x) and ...``

    >>> f = funcs_conjunction(lambda x: isinstance(x, str), lambda x: len(x) >= 5)
    >>> f('app')  # because length is less than 5...
    False
    >>> f('apple')  # length at least 5 so...
    True

    Note that in:

    >>> f(42)
    False

    it is ``False`` because it is not a string.
    This shows that the second function is not applied to the input at all, since it
    doesn't need to, and if it were, we'd get an error (length of a number?!).

    """
    return Pipe(FuncFanout(*funcs), partial(map, itemgetter(1)), all)


def funcs_disjunction(*funcs):
    """
    Makes a disjunction of functions. That is, ``func1(x) or func2(x) or ...``

    >>> f = funcs_disjunction(lambda x: x > 10, lambda x: x < -5)
    >>> f(7)
    False
    >>> f(-7)
    True
    """
    return Pipe(FuncFanout(*funcs), partial(map, itemgetter(1)), any)


def extra_wraps(func, name=None, doc_prefix=""):
    func.__name__ = name or func_name(func)
    func.__doc__ = doc_prefix + getattr(func, "__name__", "")
    return func


def mywraps(func, name=None, doc_prefix=""):
    def wrapper(wrapped):
        return extra_wraps(wraps(func)(wrapped), name=name, doc_prefix=doc_prefix)

    return wrapper


def iterize(func, name=None):
    """From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output]
    Some call this "vectorization", but it's not really a vector, but an
    iterable, thus the name.

    `iterize` is a partial of `map`.

    >>> f = lambda x: x * 10
    >>> f(2)
    20
    >>> iterized_f = iterize(f)
    >>> list(iterized_f(iter([1,2,3])))
    [10, 20, 30]

    Consider the following pipeline:

    >>> from i2 import Pipe
    >>> pipe = Pipe(lambda x: x * 2, lambda x: f"hello {x}")
    >>> pipe(1)
    'hello 2'

    But what if you wanted to use the pipeline on a "stream" of data. The
    following wouldn't work:

    >>> try:
    ...     pipe(iter([1,2,3]))
    ... except TypeError as e:
    ...     print(f"{type(e).__name__}: {e}")
    ...
    ...
    TypeError: unsupported operand type(s) for *: 'list_iterator' and 'int'

    Remember that error: You'll surely encounter it at some point.

    The solution to it is (often): ``iterize``,
    which transforms a function that is meant to be applied to a single object,
    into a function that is meant to be applied to an array, or any iterable
    of such objects.
    (You might be familiar (if you use `numpy` for example) with the related
    concept of "vectorization",
    or [array programming](https://en.wikipedia.org/wiki/Array_programming).)


    >>> from i2 import Pipe
    >>> from meshed.util import iterize
    >>> from typing import Iterable
    >>>
    >>> pipe = Pipe(
    ...     iterize(lambda x: x * 2),
    ...     iterize(lambda x: f"hello {x}")
    ... )
    >>> iterable = pipe([1, 2, 3])
    >>> # see that the result is an iterable
    >>> assert isinstance(iterable, Iterable)
    >>> list(iterable)  # consume the iterable and gather it's items
    ['hello 2', 'hello 4', 'hello 6']
    """
    # TODO: See if partialx can be used instead
    wrapper = mywraps(
        func, name=name, doc_prefix=f"generator version of {func_name(func)}:\n"
    )
    return wrapper(partial(map, func))


# from typing import Callable, Any
# from functools import wraps
# from i2 import Sig, name_of_obj


def my_isinstance(obj, class_or_tuple):
    """Same as builtin instance, but without position only constraint.
    Therefore, we can partialize class_or_tuple:

    Otherwise, couldn't do:

    >>> isinstance_of_str = partial(my_isinstance, class_or_tuple=str)
    >>> isinstance_of_str('asdf')
    True
    >>> isinstance_of_str(3)
    False

    """
    return isinstance(obj, class_or_tuple)


def instance_checker(class_or_tuple):
    """Makes a boolean function that checks the instance of an object

    >>> isinstance_of_str = instance_checker(str)
    >>> isinstance_of_str('asdf')
    True
    >>> isinstance_of_str(3)
    False

    """
    return partial(my_isinstance, class_or_tuple=class_or_tuple)


class ConditionalIterize:
    """A decorator that "iterizes" a function call if input satisfies a condition.
    That is, apply ``map(func, input)`` (iterize) or ``func(input)`` according to some
    conidition on ``input``.

    >>> def foo(x, y=2):
    ...     return x * y

    The function does this:

    >>> foo(3)
    6
    >>> foo('string')
    'stringstring'

    The iterized version of the function does this:

    >>> iterized_foo = iterize(foo)
    >>> list(iterized_foo([1, 2, 3]))
    [2, 4, 6]

    >>> from typing import Iterable
    >>> new_foo = ConditionalIterize(foo, Iterable)
    >>> new_foo(3)
    6
    >>> list(new_foo([1, 2, 3]))
    [2, 4, 6]

    See what happens if we do this:

    >>> list(new_foo('string'))
    ['ss', 'tt', 'rr', 'ii', 'nn', 'gg']

    Maybe you expected `'stringstring'` because you are thinking of `string` as a valid,
    single input. But the condition of iterization is to be an Iterable, which a
    string is, thus the (perhaps) unexpected result.

    In fact, this problem is a general one:
    If your base function doesn't process iterables, the ``isinstance(x, Iterable)``
    is good enough -- but if it is supposed to process an iterable in the first place,
    how can you distinguish whether to use the iterized version or not?
    The solution depends on the situation and the iterface you want. You choose.

    Since the situation where you'll want to iterize functions in the first place is when
    you're building streaming pipelines, a good fallback choice is to iterize if and
    only if the input is an iterator. This is condition will trigger the iterization
    when the input has a ``__next__`` -- so things like generators, but not lists,
    tuples, sets, etc.

    See in the following that ``ConditionalIterize`` also has a ``wrap`` class method
    that can be used to wrap a function at definition time.

    >>> @ConditionalIterize.wrap(Iterator)  # Iterator is the default, so no need here
    ... def foo(x, y=2):
    ...     return x * y
    >>> foo(3)
    6
    >>> foo('string')
    'stringstring'

    If you want to process a "stream" of numbers 1, 2, 3, don't do it this way:

    >>> foo([1, 2, 3])
    [1, 2, 3, 1, 2, 3]

    Instead, you should explicitly wrap that iterable in an iterator, to trigger the
    iterization:

    >>> list(foo(iter([1, 2, 3])))
    [2, 4, 6]

    So far, the only way we controlled the iterize condition is through a type.
    Really, the condition that is used behind the scenes is
    ``isinstance(obj, self.iterize_type)``.
    If you need more complex conditions though, you can specify it through the
    ``iterize_condition`` argument. The ``iterize_type`` is also used to
    annotate the resulting wrapped function if it's first argument is annotated.
    As a consequence, ``iterize_type`` needs to be a "generic" type.

    >>> @ConditionalIterize.wrap(Iterable, lambda x: isinstance(x, (list, tuple)))
    ... def foo(x: int, y=2):
    ...     return x * y
    >>> foo(3)
    6
    >>> list(foo([1, 2, 3]))
    [2, 4, 6]
    >>> from inspect import signature

    We annotated ``x`` as ``int``, so see now the annotation of the wrapped function:

    >>> str(signature(foo))
    '(x: Union[int, Iterable[int]], y=2)'

    """

    def __init__(
        self,
        func: Callable,
        iterize_type: type = Iterator,
        iterize_condition: Optional[Callable[[Any], bool]] = None,
    ):
        """

        :param func:
        :param iterize_type: The generic type to use for the new annotation
        :param iterize_condition: The condition to use to check if we should use
            the iterized version or not. If not given, will use
            ``functools.partial(my_isinstance, iterize_type)``
        """
        self.func = func
        self.iterize_type = iterize_type
        if iterize_condition is None:
            iterize_condition = instance_checker(iterize_type)
        self.iterize_condition = iterize_condition
        self.iterized_func = iterize(self.func)
        self.sig = Sig(self.func)
        wraps(self.func)(self)
        self.__signature__ = self._new_sig()

    def __call__(self, *args, **kwargs):
        _kwargs = self.sig.map_arguments(
            args, kwargs, apply_defaults=True, allow_partial=True
        )
        first_arg = next(iter(_kwargs.values()))
        if self.iterize_condition(first_arg):
            return self.iterized_func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    def __repr__(self):
        return f"<ConditionalIterize {name_of_obj(self)}{Sig(self)}>"

    def _new_sig(self):
        if len(self.sig.names) == 0:
            raise TypeError(
                f"You can only apply conditional iterization on functions that have "
                f"at least one input. This one had none: {self.func}"
            )
        first_param = self.sig.names[0]
        new_sig = self.sig  # same sig by default
        if first_param in self.sig.annotations:
            obj_annot = self.sig.annotations[first_param]
            new_sig = self.sig.ch_annotations(
                **{first_param: Union[obj_annot, self.iterize_type[obj_annot]]}
            )
        return new_sig

    @classmethod
    def wrap(
        cls,
        iterize_type: type = Iterator,
        iterize_condition: Optional[Callable[[Any], bool]] = None,
    ):
        return partial(
            cls, iterize_type=iterize_type, iterize_condition=iterize_condition
        )


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


def incremental_str_maker(str_format="{:03.f}"):
    """Make a function that will produce a (incrementally) new string at every call."""
    i = 0

    def mk_next_str():
        nonlocal i
        i += 1
        return str_format.format(i)

    return mk_next_str


lambda_name = incremental_str_maker(str_format="lambda_{:03.0f}")
unnameable_func_name = incremental_str_maker(str_format="unnameable_func_{:03.0f}")

FunctionNamer = Callable[[Callable], str]

func_name: FunctionNamer


def func_name(func) -> str:
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing
    """
    try:
        name = func.__name__
        if name == "<lambda>":
            return lambda_name()
        return name
    except AttributeError:
        return unnameable_func_name()


# ---------------------------------------------------------------------------------------
# Misc

from typing import Iterable, Callable, Optional


def args_funcnames(
    funcs: Iterable[Callable], name_of_func: Optional[FunctionNamer] = func_name
):
    """Generates (arg_name, func_id) pairs from the iterable of functions"""
    from inspect import signature, Parameter

    for func in funcs:
        sig = signature(func)
        for param in sig.parameters.values():
            arg_name = ""  # initialize
            if param.kind == Parameter.VAR_POSITIONAL:
                arg_name += "*"
            elif param.kind == Parameter.VAR_KEYWORD:
                arg_name += "**"
            arg_name += param.name  # append name of param
            yield arg_name, name_of_func(func)


def funcs_to_digraph(funcs, graph=None):
    from graphviz import Digraph

    graph = graph or Digraph()
    graph.edges(list(args_funcnames(funcs)))
    graph.body.extend([", ".join(func.__name__ for func in funcs) + " [shape=box]"])
    return graph


def dot_to_ascii(dot: str, fancy: bool = True):
    """Convert a dot string to an ascii rendering of the diagram.

    Needs a connection to the internet to work.


    >>> graph_dot = '''
    ...     graph {
    ...         rankdir=LR
    ...         0 -- {1 2}
    ...         1 -- {2}
    ...         2 -> {0 1 3}
    ...         3
    ...     }
    ... '''
    >>>
    >>> graph_ascii = dot_to_ascii(graph_dot)  # doctest: +SKIP
    >>>
    >>> print(graph_ascii)  # doctest: +SKIP
    <BLANKLINE>
                     ┌─────────┐
                     ▼         │
         ┌───┐     ┌───┐     ┌───┐     ┌───┐
      ┌▶ │ 0 │ ─── │ 1 │ ─── │   │ ──▶ │ 3 │
      │  └───┘     └───┘     │   │     └───┘
      │    │                 │   │
      │    └──────────────── │ 2 │
      │                      │   │
      │                      │   │
      └───────────────────── │   │
                             └───┘
    <BLANKLINE>

    """
    import requests

    url = "https://dot-to-ascii.ggerganov.com/dot-to-ascii.php"
    boxart = 0

    # use nice box drawing char instead of + , | , -
    if fancy:
        boxart = 1

    stripped_dot_str = dot.strip()
    if not (
        stripped_dot_str.startswith("graph") or stripped_dot_str.startswith("digraph")
    ):
        dot = "graph {\n" + dot + "\n}"

    params = {
        "boxart": boxart,
        "src": dot,
    }

    try:
        response = requests.get(url, params=params).text
    except requests.exceptions.ConnectionError:
        return "ConnectionError: You need the internet to convert dot into ascii!"

    if response == "":
        raise SyntaxError("DOT string is not formatted correctly")

    return response


def print_ascii_graph(funcs):
    digraph = funcs_to_digraph(funcs)
    dot_str = "\n".join(map(lambda x: x[1:], digraph.body[:-1]))
    print(dot_to_ascii(dot_str))


class ValidationError(ValueError):
    """Error that is raised when an object's validation failed"""


class NotUniqueError(ValidationError):
    """Error to be raised when unicity is expected, but violated"""


class NotFound(ValidationError):
    """To be raised when something is expected to exist, but doesn't"""


class NameValidationError(ValueError):
    """Use to indicate that there's a problem with a name or generating a valid name"""


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


def mk_func_name(func, exclude_names=()):
    """Makes a function name that doesn't clash with the exclude_names iterable.
    Tries it's best to not be lazy, but instead extract a name from the function
    itself."""
    name = name_of_obj(func) or "func"
    if name == "<lambda>":
        name = lambda_name()  # make a lambda name that is a unique identifier
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
                    f"{func_name}__{name}", _exclude_names
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


def _place_holder_func(*args, _sig=None, **kwargs):
    _kwargs = _sig.map_arguments(args, kwargs)
    _kwargs_str = ", ".join(f"{k}={v}" for k, v in _kwargs.items())
    return f"{_sig.name}({_kwargs_str})"


def mk_place_holder_func(arg_names_or_sig, name=None, defaults=(), annotations=()):
    """Make (working and picklable) function with a specific signature.

    This is useful for testing as well as injecting compliant functions in DAG templates.

    :param arg_names_or_sig: Anything that i2.Sig can accept as it's first input.
        (Such as a string of argument(s), function, signature, etc.)
    :param name: The ``__name__`` to give the function.
    :param defaults: If you want to add/change defaults
    :param annotations: If you want to add/change annotations
    :return: A (working and picklable) function with a specific signature


    >>> f = mk_place_holder_func('a b', 'my_func')
    >>> f(1,2)
    'my_func(a=1, b=2)'

    The first argument can be any expression of a signature that ``i2.Sig`` can
    understand. For instance, it could be a function itself.
    See how the function takes on ``mk_place_holder_func``'s signature and name in the
    following example:

    >>> g = mk_place_holder_func(mk_place_holder_func)
    >>> from inspect import signature
    >>> str(signature(g))  # should give the same signature as mk_place_holder_func
    '(arg_names_or_sig, name=None, defaults=(), annotations=())'
    >>> g(1,2,defaults=3, annotations=4)
    'mk_place_holder_func(arg_names_or_sig=1, name=2, defaults=3, annotations=4)'

    """
    defaults = dict(defaults)
    sig = Sig(arg_names_or_sig)
    sig = sig.ch_defaults(**dict(defaults))
    sig = sig.ch_annotations(**dict(annotations))

    sig.name = name or sig.name or "place_holder_func"

    func = sig(partial(_place_holder_func, _sig=sig))
    func.__name__ = sig.name

    return func


# TODO: Probably can improve efficiency and reusability using generators?
def ordered_set_operations(a: Iterable, b: Iterable) -> Tuple[List, List, List]:
    """
    Returns a triple (a-b, a&b, b-a) for two iterables a and b.
    The operations are performed as if a and b were sets, but the order in a is conserved.

    >>> ordered_set_operations([1, 2, 3, 4], [3, 4, 5, 6])
    ([1, 2], [3, 4], [5, 6])

    >>> ordered_set_operations("abcde", "cdefg")
    (['a', 'b'], ['c', 'd', 'e'], ['f', 'g'])

    >>> ordered_set_operations([1, 2, 2, 3], [2, 3, 3, 4])
    ([1], [2, 3], [4])
    """
    set_b = set(b)
    a = tuple(a)  # because a traversed three times (consider a one-pass algo)
    a_minus_b = [x for x in a if x not in set_b]
    a_intersect_b = [x for x in a if x in set_b and not set_b.remove(x)]
    b_minus_a = [x for x in b if x not in set(a)]

    return a_minus_b, a_intersect_b, b_minus_a


# utils to reorder funcnodes


def pairs(xs):
    if len(xs) <= 1:
        return xs
    else:
        pairs = list(zip(xs, xs[1:]))
    return pairs


def curry(func):
    def res(*args):
        return func(tuple(args))

    return res


def uncurry(func):
    def res(tup):
        return func(*tup)

    return res


Renamer = Union[Callable[[str], str], str, Mapping[str, str]]


def _if_none_return_input(func):
    """Wraps a function so that when the original func outputs None, the wrapped will
    return the original input instead.

    >>> def func(x):
    ...     if x % 2 == 0:
    ...         return None
    ...     else:
    ...         return x * 10
    >>> wfunc = _if_none_return_input(func)
    >>> func(3)
    30
    >>> wfunc(3)
    30
    >>> assert func(4) is None
    >>> wfunc(4)
    4
    """

    def _func(input_val):
        if (output_val := func(input_val)) is not None:
            return output_val
        else:
            return input_val

    return _func


def numbered_suffix_renamer(name, sep="_"):
    """
    >>> numbered_suffix_renamer('item')
    'item_1'
    >>> numbered_suffix_renamer('item_1')
    'item_2'
    """
    p = re.compile(sep + r"(\d+)$")
    m = p.search(name)
    if m is None:
        return f"{name}{sep}1"
    else:
        num = int(m.group(1)) + 1
        return p.sub(f"{sep}{num}", name)


class InvalidFunctionParameters(ValueError):
    """To be used when a function's parameters are not compliant with some rule about
    them."""


def _suffix(start=0):
    i = start
    while True:
        yield f"_{i}"
        i += 1


def _add_suffix(x, suffix):
    return f"{x}{suffix}"


incremental_suffixes = _suffix()
_renamers = (lambda x: f"{x}{suffix}" for suffix in incremental_suffixes)


def _return_val(first_arg, val):
    return val


def _equality_checker(x, val):
    return x == val


def _not_callable(obj):
    return not callable(obj)


# Pattern: routing
# TODO: Replace
def conditional_trans(
    obj: T, condition: Callable[[T], bool], trans: Callable[[T], Any]
):
    """Conditionally transform an object unless it is marked as a literal.

    >>> from functools import partial
    >>> trans = partial(
    ...     conditional_trans, condition=str.isnumeric, trans=float
    ... )
    >>> trans('not a number')
    'not a number'
    >>> trans('10')
    10.0

    To use this function but tell it to not transform some a specific input no matter
    what, wrap the input with ``Literal``

    >>> # from meshed import Literal
    >>> conditional_trans(LiteralVal('10'), str.isnumeric, float)
    '10'

    """
    # TODO: Maybe make Literal checking less sensitive to isinstance checks, using
    #   hasattr instead for example.
    if isinstance(obj, LiteralVal):  # If val is a Literal, return its value as is
        return obj.val
    elif condition(obj):  # If obj satisfies condition, return the alternative_obj
        return trans(obj)
    else:  # If not, just return object
        return obj


def replace_item_in_iterable(iterable, condition, replacement, *, egress=None):
    """Returns a list where all items satisfying ``condition(item)`` were replaced
    with ``replacement(item)``.

    If ``condition`` is not a callable, it will be considered as a value to check
    against using ``==``.

    If ``replacement`` is not a callable, it will be considered as the actual
    value to replace by.

    :param iterable: Input iterable of items
    :param condition: Condition to apply to item to see if it should be replaced
    :param replacement: (Conditional) replacement value or function
    :param egress: The function to apply to transformed iterable

    >>> replace_item_in_iterable([1,2,3,4,5], condition=2, replacement = 'two')
    [1, 'two', 3, 4, 5]
    >>> is_even = lambda x: x % 2 == 0
    >>> replace_item_in_iterable([1,2,3,4,5], condition=is_even, replacement = 'even')
    [1, 'even', 3, 'even', 5]
    >>> replace_item_in_iterable([1,2,3,4,5], is_even, replacement=lambda x: x * 10)
    [1, 20, 3, 40, 5]

    Note that if the input iterable is not a ``list``, ``tuple``, or ``set``,
    your output will be an iterator that you'll have to iterate through to gather
    transformed items.

    >>> g = replace_item_in_iterable(iter([1,2,3,4,5]), condition=2, replacement = 'two')
    >>> isinstance(g, Iterator)
    True

    Unless you specify an egress of your choice:

    >>> replace_item_in_iterable(
    ... iter([1,2,3,4,5]), is_even, lambda x: x * 10, egress=sorted
    ... )
    [1, 3, 5, 20, 40]

    """
    # If condition or replacement are not callable, make them so
    condition = conditional_trans(
        condition, _not_callable, lambda val: partial(_equality_checker, val=val)
    )
    replacement = conditional_trans(
        replacement, _not_callable, lambda val: partial(_return_val, val=val)
    )
    # Handle the egress argument
    if egress is None:
        if isinstance(iterable, (list, tuple, set)):
            egress = type(iterable)
        else:
            egress = lambda x: x  # that is return "as is"

    # Make the item replacer
    item_replacer = partial(conditional_trans, condition=condition, trans=replacement)

    return egress(map(item_replacer, iterable))


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


def inverse_dict_asserting_losslessness(d: dict):
    inv_d = {v: k for k, v in d.items()}
    assert len(inv_d) == len(d), (
        f"can't invert: You have some duplicate values in this dict: " f"{d}"
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

    >>> extract_values({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    (1, 3)

    Order matters!

    >>> extract_values({'a': 1, 'b': 2, 'c': 3}, ['c', 'a'])
    (3, 1)

    """
    tup = tuple(_extract_values(d, keys))
    if len(tup) > 1:
        return tup
    elif len(tup) == 1:
        return tup[0]
    else:
        return None


def extract_items(d: dict, keys: Iterable):
    """generator of (k, v) pairs extracted from d for keys

    >>> list(extract_items({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']))
    [('a', 1), ('c', 3)]

    """
    for k in keys:
        yield k, d[k]


def extract_dict(d: dict, keys: Iterable):
    """Extract items from dict ``d``, returning them as a dict.

    >>> extract_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    {'a': 1, 'c': 3}

    Order matters!

    >>> extract_dict({'a': 1, 'b': 2, 'c': 3}, ['c', 'a'])
    {'c': 3, 'a': 1}

    """
    return dict(extract_items(d, keys))


ParameterMerger = Callable[[Iterable[Parameter]], Parameter]
parameter_merger: ParameterMerger


# TODO: Be aware of i2.signatures.param_comparator in
#  https://github.com/i2mint/i2/blob/2bd43b350a3ae29f1e6c587dbe15d6f536635173/i2/signatures.py#L4247
#  and related funnctions, which are meant to be a more general approach. Consider
#  merging parameter_merger to use that general tooling.
# TODO: Make the ValidationError be even more specific, indicating what parameters
#  are different and how.
def parameter_merger(
    *params, same_name=True, same_kind=True, same_default=True, same_annotation=True
):
    """Validates that all the params are exactly the same, returning the first if so.

    This is used when hooking up functions that use the same parameters (i.e. arg
    names). When the name of an argument is used more than once, which kind, default,
    and annotation should be used in the interface of the DAG?

    If they're all the same, there's no problem.

    But if they're not the same, we need to provide control on which to ignore.

    >>> from inspect import Parameter as P
    >>> PK = P.POSITIONAL_OR_KEYWORD
    >>> KO = P.KEYWORD_ONLY
    >>> parameter_merger(P('a', PK), P('a', PK))
    <Parameter "a">
    >>> parameter_merger(P('a', PK), P('different_name', PK), same_name=False)
    <Parameter "a">
    >>> parameter_merger(P('a', PK), P('a', KO), same_kind=False)
    <Parameter "a">
    >>> parameter_merger(P('a', PK), P('a', PK,  default=42), same_default=False)
    <Parameter "a">
    >>> parameter_merger(P('a', PK, default=42), P('a', PK), same_default=False)
    <Parameter "a=42">
    >>> parameter_merger(P('a', PK, annotation=int), P('a', PK), same_annotation=False)
    <Parameter "a: int">
    """
    suggestion_on_error = """To resolve this you have several choices:

    - Change the properties of the param (kind, default, annotation) to be those you 
      want. For example, you can use ``i2.Sig.ch_param_attrs`` on the signatures 
      (or ``i2.Sig.ch_names``, ``i2.Sig.ch_defaults``, ``i2.Sig.ch_kinds``, 
      ``i2.Sig.ch_annotations``)
      to get a function decorator that will do that for you.
    - If you're making a DAG, consider specifying a different ``parameter_merge``.
      For example you can use ``functools.partial`` on 
      ``meshed.parameter_merger``, fixing ``same_kind``, ``same_default``, 
      and/or ``same_annotation`` to ``False`` to get a more lenient version of it.
      (See also i2.signatures.param_comparator.)

    See https://github.com/i2mint/i2/discussions/63 and 
    https://github.com/i2mint/meshed/issues/7 (description and comments) for more
    info.
    """
    first_param, *_ = params
    if same_name and not all(p.name == first_param.name for p in params):
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
            f"{params}\n{suggestion_on_error}"
        )
    return first_param


conservative_parameter_merge: ParameterMerger = partial(
    parameter_merger, same_kind=True, same_default=True, same_annotation=True
)
