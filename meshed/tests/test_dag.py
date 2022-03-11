"""Test dags"""
import pytest


def test_funcnode_bind():
    """
    Test the renaming of arguments and output of functions using FuncNode and its
    effect on DAG
    """
    from meshed.dag import DAG
    from meshed import FuncNode

    def f(a, b):
        return a + b

    def g(a_plus_b, d):
        return a_plus_b * d

    # here we specify that the output of f will be injected in g as an argument for the parameter a_plus_b
    f_node = FuncNode(func=f, out='a_plus_b')
    g_node = FuncNode(func=g)
    dag = DAG((f_node, g_node))
    assert dag(a=1, b=2, d=3) == 9

    # we can do more complex renaming as well, for example here we specify that the value for b is also the value for d,
    # resulting in the dag being now 2 variable dag
    f_node = FuncNode(func=f, out='a_plus_b')
    g_node = FuncNode(func=g, bind={'d': 'b'})
    dag = DAG((f_node, g_node))
    assert dag(a=1, b=2) == 6


def test_iterize_dag():
    def f(a, b=2):
        return a + b

    def g(f, c=3):
        return f * c

    from meshed import DAG

    d = DAG([f, g])
    # d.dot_digraph()  # smoke testing the digraph

    assert (  # if you needed to apply d to an iterator, you'd normally do this
        list(map(d, [1, 2, 3]))
    ) == ([9, 12, 15])

    # But if you need a function that "looks like" d, but is "vectorized" (really
    # iterized) version...
    from functools import partial
    from inspect import signature

    def iterize(func):
        _iterized_func = partial(map, func)
        _iterized_func.__signature__ = signature(func)
        return _iterized_func

    di = iterize(d)
    # di has the same signature as d:
    assert signature(di) == signature(d)
    assert (list(di([1, 2, 3]))) == ([9, 12, 15])  # But works with a being an iterator

    # Note that di will return an iterator that needs to be "consumed" (here with list)
    # That is, no matter what the (iterable) type of the input is.
    # If you wanted to systematically get your output as a list (or tuple, or set,
    # or numpy.array),
    # there's several choices...

    # You could use i2.Pipe

    from i2 import Pipe

    di_list = Pipe(di, list)
    assert di_list([1, 2, 3]) == [9, 12, 15]


def test_binding_to_a_root_node():
    """
    See: https://github.com/i2mint/meshed/issues/7
    """
    from meshed.dag import DAG
    from meshed.util import ValidationError
    from meshed import FuncNode

    def f(a, b):
        return a + b

    def g(a_plus_b, d):
        return a_plus_b * d

    # we bind d to b, and it works!
    f_node = FuncNode(func=f, out='a_plus_b')
    g_node = FuncNode(func=g, bind={'d': 'b'})
    dag = DAG((f_node, g_node))
    assert dag(a=1, b=2) == 6

    # but if b and d are not aligned on all other parameter props besides name
    # (kind, default, annotation), then we get an error

    def gg(a_plus_b, d=4):
        return a_plus_b * d

    gg_node = FuncNode(func=gg, bind={'d': 'b'})

    with pytest.raises(ValidationError) as e_info:
        _ = DAG((f_node, gg_node))

    assert "didn't have the same default" in e_info.value.args[0]

    # There's several solutions to this.
    # First, we can simply prepare the functions so that the defaults align.
    # The following shows how to do this in two different ways

    # 1: "Manually"
    def ff(a, b=4):
        return f(a, b)

    ff_node = FuncNode(func=ff, out='a_plus_b')
    dag = DAG((ff_node, gg_node))
    assert dag(a=1, b=2) == 6

    # 2: With i2.Sig
    from i2 import Sig

    give_default_to_b = lambda func: Sig(func).ch_defaults(b=4)(func)
    ff_node = FuncNode(func=give_default_to_b(f), out='a_plus_b')
    dag = DAG((ff_node, gg_node))
    assert dag(a=1, b=2) == 6
    # And if you don't specify b, it has that default you set!
    assert dag(a=1) == 20

    # Second, we could specify a different "merging policy" (the function that
    # determines how to resolve the issue of several params with the same name
    # (or binding) that conflict on some prop (kind, default and/or annotation)

    # Before we go there though, let's show that default is not the only problem.
    # If the annotation, or the kind are different, we also run in to the same problem
    # (and solution to it)
    def f(a, b):
        return a + b

    def ggg(a_plus_b, d: int):  # note that d has no default, but an annotation
        return a_plus_b * d

    ggg_node = FuncNode(func=ggg, bind={'d': 'b'})
    with pytest.raises(ValidationError) as e_info:
        _ = DAG((f_node, ggg_node))
    assert "didn't have the same annotation" in e_info.value.args[0]

    # Solution (with i2.Sig)

    give_annotation_to_b = lambda func: Sig(func).ch_annotations(b=int)(func)
    ff_node = FuncNode(func=give_annotation_to_b(f), out='a_plus_b')
    dag = DAG((ff_node, ggg_node))
    assert dag(a=1, b=2) == 6

    # The other solution to the parameter property misalignment is to tell the DAG
    # constructor what we want it to do with conflicts. For example, just ignore them.
    # (Not a good general policy though!)

    from meshed.dag import conservative_parameter_merge
    from functools import partial

    first_wins_all_merger = partial(
        conservative_parameter_merge,
        same_kind=False,
        same_default=False,
        same_annotation=False,
    )

    def f(a, b: int, /):
        return a + b

    def g(a_plus_b, d: float = 4):
        return a_plus_b * d

    lenient_dag_maker = partial(DAG, parameter_merge=first_wins_all_merger)

    f_node = FuncNode(func=f, out='a_plus_b')
    g_node = FuncNode(func=g, bind={'d': 'b'})
    dag = lenient_dag_maker([f_node, g_node])
    assert dag(1, 2) == 6
    # Note we can't do dag(a=1, b=2) since (like f) it's position-only.
    # Indeed the dag inherits its arguments' properties from the functions composing it, in this case f

    # Resolving conflicts this way isn't the best general policy (that's why it's not
    # the default).
    # In production, it's advised to implement a more careful merging policy, possibly
    # specifying (in the `parameter_merge` callable itself) explicitly what to do for
    # every situation that we encounter.


def test_dag_partialize():
    from functools import partial
    from i2 import Sig
    from meshed import DAG
    from inspect import signature

    def foo(a, b):
        return a - b

    f = DAG([foo])
    assert str(Sig(f)) == '(a, b)'

    # if we give ``b`` a default:
    ff = f.partial(b=9)
    assert str(Sig(ff)) == '(a, b=9)'
    # note that the Sig of the partial of foo is '(a, *, b=9)' though
    assert str(Sig(partial(foo, b=9))) == '(a, *, b=9)'
    assert ff(10) == ff(a=10) == 1

    # if we give ``a`` (the first arg) a default but not ``b`` (the second arg)
    fff = f.partial(a=4)  # fixing a, which is before b
    # note that this fixing a reorders the parameters (so we have a valid signature!)
    assert str(Sig(fff)) == '(b, a=4)'

    fn = fff.func_nodes[0]
    assert fn(dict(b=3)) == 1

    def f(a, b):
        return a + b

    def g(c, d=4):
        return c * d

    def h(f, g):
        return g - f

    larger_dag = DAG([f, g, h])

    new_dag = larger_dag.partial(c=3, a=1)
    assert new_dag(b=5, d=6) == 12
    assert str(signature(new_dag)) == '(b, a=1, c=3, d=4)'
