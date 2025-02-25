"""Objects for testing"""

from inspect import signature
from meshed.dag import DAG
from meshed import FuncNode


def f(a, b):
    return a + b


def g(a_plus_b, d):
    return a_plus_b * d


# here we specify that the output of f will be injected in g as an argument for the
# parameter a_plus_b
f_node = FuncNode(func=f, out="a_plus_b")
g_node = FuncNode(func=g)
dag_plus_and_times = DAG((f_node, g_node))
assert dag_plus_and_times(a=1, b=2, d=3) == 9


# we can do more complex renaming as well, for example here we specify that the value
# for b is also the value for d,
# resulting in the dag being now 2 variable dag
f_node = FuncNode(func=f, out="a_plus_b")
g_node = FuncNode(func=g, bind={"d": "b"})
dag_plus_and_times_ext = DAG((f_node, g_node))
assert dag_plus_and_times_ext(a=1, b=2) == 6


def f(a, b):
    return a + b


def g(c, d=4):
    return c * d


def h(ff, gg=42):
    return gg - ff


dag_plus_times_minus = DAG(
    [
        FuncNode(f, out="f_out", name="f"),
        FuncNode(g, out="g_out", name="g", func_label="The G Node"),
        FuncNode(h, bind={"ff": "f_out", "gg": "g_out"}),
    ]
)
dag_plus_times_minus.__doc__ = """
A three node DAG with a variety of artifacts 
(non-default out, bind, and func_label, as well as
a defaulted root node and a defaulted middle node)
"""

dag_plus_times_minus_partial = dag_plus_times_minus.partial(c=3, a=1)
assert dag_plus_times_minus_partial(b=5, d=6) == 12
assert str(signature(dag_plus_times_minus_partial)) == "(b, a=1, c=3, d=4)"
