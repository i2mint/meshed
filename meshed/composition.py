"""Specific use of FuncNode and DAG"""

from inspect import signature
from meshed import FuncNode, DAG


def get_param(func):
    """
    Find the name of the parameter of a function with exactly one parameter. Raise an error if more or less parameters.
    :param func: callable, the function to inspect
    :return: str, the name of the single parameter of func
    """

    params = signature(func).parameters.keys()
    assert (
        len(params) == 1
    ), f"Your function has more than 1 parameter! Namely: {', '.join(params)}"
    for param in params:
        return param


def line_with_dag(*steps):
    """
    Emulate a Line object with a DAG
    :param steps: an iterable of callables, the steps of the pipeline. Each step should have exactly one parameter
    and the output of each step is fed into the next
    :return: a DAG instance computing the composition of all the functions in steps, in the provided order
    """

    step_counter = 0
    first_node = FuncNode(steps[0], out=f'step_{step_counter}')
    func_nodes = [first_node]
    for step in steps[1:]:
        step_node = FuncNode(
            step,
            out=f'step_{step_counter + 1}',
            bind={get_param(step): f'step_{step_counter}'},
        )
        step_counter += 1
        func_nodes.append(step_node)

    return DAG(func_nodes)
