"""util functions"""


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


def name_of_obj(o):
    """
    Tries to find the (or "a") name for an object, even if `__name__` doesn't exist.

    >>> name_of_obj(map)
    'map'
    >>> name_of_obj([1, 2, 3])
    'list'
    >>> name_of_obj(print)
    'print'
    >>> name_of_obj(lambda x: x)
    '<lambda>'
    >>> from functools import partial
    >>> name_of_obj(partial(print, sep=','))
    'print'
    """
    if hasattr(o, "__name__"):
        return o.__name__
    elif hasattr(o, "__class__"):
        name = name_of_obj(o.__class__)
        if name == "partial":
            if hasattr(o, "func"):
                return name_of_obj(o.func)
        return name
    else:
        return None


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


def func_name(func):
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing"""
    try:
        name = func.__name__
        if name == "<lambda>":
            return lambda_name()
        return name
    except AttributeError:
        return unnameable_func_name()
