"""Ideas on collapsing and expanding nodes
See "Collapse and expand nodes" discussion: 
https://github.com/i2mint/meshed/discussions/54

"""

from dataclasses import dataclass
from i2 import Sig
from meshed.dag import DAG


@dataclass
class CollapsedDAG:
    """To collapse a DAG into a single function

    This is useful for when you want to use a DAG as a function,
    but you don't want to see all the arguments.

    """

    dag: DAG

    def __post_init__(self):
        Sig(self.dag)(self)  # so that __call__ gets dag's signature
        self.__name__ = self.dag.name

    def __call__(self, *args, **kwargs):
        return self.dag(*args, **kwargs)

    def expand(self):
        return self.dag


# TODO: Finish this
def expand_nodes(
    dag,
    nodes=None,
    *,
    is_node=lambda fnode, node: fnode.name == node or fnode.out == node,
    expansion_record_store=None,  # TODO: Implement this to keep track of what was expanded
):
    if nodes is None:
        nodes = ...  # find all func_nodes that have isinstance(fn.func, CollapsedDAG)

    def change_node_or_not(node):
        if is_node(node):
            return CollapsedDAG(node.func)
        else:
            return node

    return DAG(list(map(change_node_or_not, dag.func_nodes)))
