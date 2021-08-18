"""Caching meshes"""

from functools import cached_property
from inspect import signature

from meshed.util import func_name


class Literal:
    """An object to indicate that the value should be considered literally"""

    def __init__(self, val):
        self.val = val


def set_cached_property_attr(obj, name, value):
    cached_value = cached_property(value)
    cached_value.__set_name__(obj, name)
    setattr(obj, name, cached_value)


class LazyProps:
    """A class that makes all"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        #         cls.__literals = []
        #         cls.__lazyprops = []
        for attr_name in (a for a in dir(cls) if not a.startswith('__')):
            attr_obj = getattr(cls, attr_name)
            if isinstance(attr_obj, Literal):
                setattr(cls, attr_name, attr_obj.val)
            #                 cls._LazyProps__literals.append(attr_name)
            else:
                set_cached_property_attr(cls, attr_name, attr_obj)

    #                 cls._LazyProps__lazyprops.append(attr_name)

    Literal = Literal  # just to have Literal available as LazyProps.Literal


def add_cached_property(cls, method, attr_name=None):
    attr_name = attr_name or func_name(method)
    set_cached_property_attr(cls, attr_name, method)
    return cls


def add_cached_property_from_func(cls, func, attr_name=None):
    params = list(signature(func).parameters)

    def method(self):
        return func(**{k: getattr(self, k) for k in params})

    method.__name__ = func.__name__
    method.__doc__ = func.__doc__

    return add_cached_property(cls, method, attr_name)


def with_cached_properties(funcs):
    def add_cached_properties(cls):
        for func in funcs:
            if not callable(func):
                func, attr_name = func  # assume it's a (func, attr_name) pair
            else:
                attr_name = None
            add_cached_property_from_func(cls, func, attr_name)
        return cls

    return add_cached_properties
