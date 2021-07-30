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
    if hasattr(o, '__name__'):
        return o.__name__
    elif hasattr(o, '__class__'):
        name = name_of_obj(o.__class__)
        if name == 'partial':
            if hasattr(o, 'func'):
                return name_of_obj(o.func)
        return name
    else:
        return None
