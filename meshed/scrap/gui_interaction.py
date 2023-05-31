"""
This module contains some ideas around making a two-way interaction between meshed 
and a GUI that will enable the construction of meshes as well as rendering them, 
and possibly running them.
"""

from typing import Callable
from meshed.dag import DAG, FuncNode
from meshed.util import mk_place_holder_func
from functools import partial

# TODO: Add default func_to_jdict and dict_to_func that uses mk_place_holder_func to
#  jdict will be only signature (jdict) and deserializing it will be just placeholder
Jdict = dict  # json-serializable dictionary


def fnode_to_jdict(
    fnode: FuncNode, *, func_to_jdict: Callable[[Callable], Jdict] = None
):
    jdict = {
        'name': fnode.name,
        'func_label': fnode.func_label,
        'bind': fnode.bind,
        'out': fnode.out,
    }
    if func_to_jdict is not None:
        jdict['func'] = func_to_jdict(fnode.func)
    return jdict


def jdict_to_fnode(jdict: dict, *, jdict_to_func: Callable[[Jdict], Callable] = None):
    fnode = FuncNode(
        name=jdict['name'],
        func_label=jdict['func_label'],
        bind=jdict['bind'],
        out=jdict['out'],
    )
    if jdict_to_func is not None:
        fnode.func = jdict_to_func(jdict['func'])
    return fnode


def dag_to_jdict(dag: DAG, *, func_to_jdict: Callable = None):
    """
    Will produce a json-serializable dictionary from a dag.
    """
    fnode_to_jdict_ = partial(fnode_to_jdict, func_to_jdict=func_to_jdict)
    return {
        'name': dag.name,
        'func_nodes': list(map(fnode_to_jdict_, dag.func_nodes)),
    }


def jdict_to_dag(jdict: dict, *, jdict_to_func: Callable = None):
    """
    Will produce a dag from a json-serializable dictionary.
    """
    jdict_to_fnode_ = partial(jdict_to_fnode, jdict_to_func=jdict_to_func)
    return DAG(
        name=jdict['name'], func_nodes=list(map(jdict_to_fnode_, jdict['func_nodes'])),
    )
