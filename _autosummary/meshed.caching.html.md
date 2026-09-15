# meshed.caching

Turn functions into cached properties of a class.

The functions here attach `functools.cached_property` attributes to a class
after the fact, so a value is computed once per instance, on first access, from
other attributes of that instance. `LazyProps` does this for every
one-argument callable of a subclass at class-creation time; the
`with_cached_properties` decorator does it for a chosen list of functions,
sourcing each function’s parameters from same-named instance attributes.

Main entry points:

- `LazyProps`: base class that makes each one-argument method a cached property.
- `with_cached_properties`: class decorator adding the given functions as cached
  properties.
- `add_cached_property_from_func`: the same for a single function, without decorating.

```pycon
>>> from meshed.caching import with_cached_properties
>>> def area(width, height):
...     print('computing area')
...     return width * height
>>> @with_cached_properties([area])
... class Rect:
...     def __init__(self, width, height):
...         self.width, self.height = width, height
>>> r = Rect(2, 3)
>>> r.area
computing area
6
>>> r.area
6
```

### Functions

| [`add_cached_property`](#meshed.caching.add_cached_property)(cls, method[, attr_name])   | Add a method as a cached property to a class.    |
|--------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [`add_cached_property_from_func`](#meshed.caching.add_cached_property_from_func)(cls, func[, ...]) | Add a function cached property to a class.       |
| [`set_cached_property_attr`](#meshed.caching.set_cached_property_attr)(obj, name, value)      | Helper to set cached properties.                 |
| [`with_cached_properties`](#meshed.caching.with_cached_properties)(funcs)                   | A decorator to add cached properties to a class. |

### Classes

| [`LazyProps`](#meshed.caching.LazyProps)()   | A class that makes all its attributes cached_property properties.   |
|----------------------------------------------------------------|---------------------------------------------------------------------|

### *class* meshed.caching.LazyProps

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A class that makes all its attributes cached_property properties.

### Example

```pycon
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
```

#### Literal

alias of `LiteralVal`

### meshed.caching.add_cached_property(cls, method, attr_name=None)

Add a method as a cached property to a class.

### meshed.caching.add_cached_property_from_func(cls, func, attr_name=None)

Add a function cached property to a class.

### meshed.caching.set_cached_property_attr(obj, name, value)

Helper to set cached properties.

Reason: When adding cached_property dynamically (not just with the @cached_property)
the name is not set correctly. This solves that.

### meshed.caching.with_cached_properties(funcs)

A decorator to add cached properties to a class.
