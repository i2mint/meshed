"""util functions"""
from functools import partial, wraps
from typing import Callable, Any, Union, Iterator, Optional, Iterable

from i2 import Sig, name_of_obj


def extra_wraps(func, name=None, doc_prefix=''):
    func.__name__ = name or func_name(func)
    func.__doc__ = doc_prefix + getattr(func, '__name__', '')
    return func


def mywraps(func, name=None, doc_prefix=''):
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
        func, name=name, doc_prefix=f'generator version of {func_name(func)}:\n'
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
    """

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
        _kwargs = self.sig.kwargs_from_args_and_kwargs(
            args, kwargs, apply_defaults=True, allow_partial=True
        )
        first_arg = next(iter(_kwargs.values()))
        if self.iterize_condition(first_arg):
            return self.iterized_func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    def __repr__(self):
        return f'<ConditionalIterize {name_of_obj(self)}{Sig(self)}>'

    def _new_sig(self):
        if len(self.sig.names) == 0:
            raise TypeError(
                f'You can only apply conditional iterization on functions that have '
                f'at least one input. This one had none: {self.func}'
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


def incremental_str_maker(str_format='{:03.f}'):
    """Make a function that will produce a (incrementally) new string at every call."""
    i = 0

    def mk_next_str():
        nonlocal i
        i += 1
        return str_format.format(i)

    return mk_next_str


lambda_name = incremental_str_maker(str_format='lambda_{:03.0f}')
unnameable_func_name = incremental_str_maker(str_format='unnameable_func_{:03.0f}')

FunctionNamer = Callable[[Callable], str]

func_name: FunctionNamer


def func_name(func) -> str:
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing"""
    try:
        name = func.__name__
        if name == '<lambda>':
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
            arg_name = ''  # initialize
            if param.kind == Parameter.VAR_POSITIONAL:
                arg_name += '*'
            elif param.kind == Parameter.VAR_KEYWORD:
                arg_name += '**'
            arg_name += param.name  # append name of param
            yield arg_name, name_of_func(func)


def funcs_to_digraph(funcs, graph=None):
    from graphviz import Digraph

    graph = graph or Digraph()
    graph.edges(list(args_funcnames(funcs)))
    graph.body.extend([', '.join(func.__name__ for func in funcs) + ' [shape=box]'])
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

    url = 'https://dot-to-ascii.ggerganov.com/dot-to-ascii.php'
    boxart = 0

    # use nice box drawing char instead of + , | , -
    if fancy:
        boxart = 1

    stripped_dot_str = dot.strip()
    if not (
        stripped_dot_str.startswith('graph') or stripped_dot_str.startswith('digraph')
    ):
        dot = 'graph {\n' + dot + '\n}'

    params = {
        'boxart': boxart,
        'src': dot,
    }

    try:
        response = requests.get(url, params=params).text
    except requests.exceptions.ConnectionError:
        return 'ConnectionError: You need the internet to convert dot into ascii!'

    if response == '':
        raise SyntaxError('DOT string is not formatted correctly')

    return response


def print_ascii_graph(funcs):
    digraph = funcs_to_digraph(funcs)
    dot_str = '\n'.join(map(lambda x: x[1:], digraph.body[:-1]))
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
            name = f'{prefix}__{i}'
            if name not in exclude_names:
                return name
            i += 1


def mk_func_name(func, exclude_names=()):
    """Makes a function name that doesn't clash with the exclude_names iterable.
    Tries it's best to not be lazy, but instead extract a name from the function
    itself."""
    name = name_of_obj(func) or 'func'
    if name == '<lambda>':
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


def _place_holder_func(*args, _sig=None, **kwargs):
    _kwargs = _sig.kwargs_from_args_and_kwargs(args, kwargs)
    _kwargs_str = ', '.join(f'{k}={v}' for k, v in _kwargs.items())
    return f'{_sig.name}({_kwargs_str})'


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

    sig.name = name or sig.name or 'place_holder_func'

    func = sig(partial(_place_holder_func, _sig=sig))
    func.__name__ = sig.name

    return func
