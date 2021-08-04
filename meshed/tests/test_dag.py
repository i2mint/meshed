"""Test dags"""


def iterize_dag_test():
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
    from lined import iterize
    from i2 import Sig

    di = iterize(d)
    # di has the same signature as d:
    Sig(di)
    assert (list(di([1, 2, 3]))) == ([9, 12, 15])  # But works with a being an iterator

    # Note that di will return an iterator that needs to be "consumed" (here with list)
    # That is, no matter what the (iterable) type of the input is.
    # If you wanted to systematically get your output as a list (or tuple, or set,
    # or numpy.array),
    # there's several choices...

    # You could use lined.Line

    from lined import Line

    di_list = Line(di, list)
    assert di_list([1, 2, 3]) == [9, 12, 15]
