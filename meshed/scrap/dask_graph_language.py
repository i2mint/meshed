"""How to make dags from the dask specification

See https://docs.dask.org/en/latest/graphs.html#example for the specification
"""


from meshed import FuncNode, DAG
from i2 import Sig


def node_funcs_from_dask_graph_dict(dask_graph_dict):
    for out_key, val in dask_graph_dict.items():
        if isinstance(val, tuple):
            func, *args = val
            bind_args = dict(zip(Sig(func).names, args))
            yield FuncNode(func=func, bind=bind_args, out=out_key)


def inc(i):
    return i + 1


def add(a, b):
    return a + b


d = {'y': (inc, 'x'), 'z': (add, 'y', 'a')}

dag = DAG(node_funcs_from_dask_graph_dict(d))

from contextlib import suppress

with suppress(ModuleNotFoundError, ImportError):
    dag.dot_digraph()


# For being able to handle non-tuple vals (like the 1 of 'x':1 and non string args
# (like the 10)), need more work.
# Can solve ambiguity between string INPUT and string denoting scope argument name with
# a AsString literal class
# d = {'x': 1,
#      'y': (inc, 'x'),
#      'z': (add, 'y', 10)}
