"""
`meshed` contains a set of tools that allow the developer to provide a collection
of python objects (think functions) and some policy of how these should be connected
and get an aggregate object that will use the underlying objects in some way.

If you want something concrete, think of the python objects to be functions,
and the aggregation policies to be things like "function composition" (pipelines)
or DAGs.
But the intent is to be able to get more general aggregations than those.

Extras
------

`itools.py` contain tools that enable operations on graphs where graphs are represented
by an adjacency Mapping.

"""


from meshed.dag import DAG
from meshed.base import FuncNode
from meshed.makers import code_to_dag
