"""Caching meshes"""

from functools import cached_property
from inspect import signature

from meshed.util import func_name, LiteralVal


def set_cached_property_attr(obj, name, value):
    """
    Helper to set cached properties.

    Reason: When adding cached_property dynamically (not just with the @cached_property)
    the name is not set correctly. This solves that.
    """
    cached_value = cached_property(value)
    cached_value.__set_name__(obj, name)
    setattr(obj, name, cached_value)


class LazyProps:
    """
    A class that makes all its attributes cached_property properties.

    Example:

    >>> class Klass(LazyProps):
    ...     a = 1
    ...     b = 2
    ...
    ...     # methods with one argument are cached
    ...     def c(self):
    ...         print("computing c...")
    ...         return self.a + self.b
    ...
    ...     d = lambda x: 4
    ...     e = LazyProps.Literal(lambda x: 4)
    ...
    ...     @LazyProps.Literal  # to mark that this method should not be cached
    ...     def method1(self):
    ...         return self.a * 7
    ...
    ...     # Methods with more than one argument are not cached
    ...     def method2(self, x):
    ...         return x + 1
    ...
    ...
    >>> k = Klass()
    >>> k.b
    2
    >>> k.c
    computing c...
    3
    >>> k.c  # note that c is not recomputed
    3
    >>> k.d  # d, a lambda with one argument, is treated as a cached property
    4
    >>> k.e()  # e is marked as a literal so is not a cached property, so need to call
    4
    >>> k.method1()  # method1 has one argument, but marked as a literal
    7
    >>> k.method2(10)  # method2 has more than one argument, so is not a cached property
    11
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for attr_name in (a for a in dir(cls) if not a.startswith("__")):
            attr_obj = getattr(cls, attr_name)
            if isinstance(attr_obj, LiteralVal):
                setattr(cls, attr_name, attr_obj.val)
            elif callable(attr_obj) and len(signature(attr_obj).parameters) == 1:
                set_cached_property_attr(cls, attr_name, attr_obj)

    Literal = LiteralVal  # just to have Literal available as LazyProps.Literal


def add_cached_property(cls, method, attr_name=None):
    """
    Add a method as a cached property to a class.
    """
    attr_name = attr_name or func_name(method)
    set_cached_property_attr(cls, attr_name, method)
    return cls


def add_cached_property_from_func(cls, func, attr_name=None):
    """
    Add a function cached property to a class.
    """
    params = list(signature(func).parameters)

    def method(self):
        return func(**{k: getattr(self, k) for k in params})

    method.__name__ = func.__name__
    method.__doc__ = func.__doc__

    return add_cached_property(cls, method, attr_name)


def with_cached_properties(funcs):
    """
    A decorator to add cached properties to a class.
    """

    def add_cached_properties(cls):
        for func in funcs:
            if not callable(func):
                func, attr_name = func  # assume it's a (func, attr_name) pair
            else:
                attr_name = None
            add_cached_property_from_func(cls, func, attr_name)
        return cls

    return add_cached_properties
