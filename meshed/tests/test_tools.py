"""Test hybrid dag that uses a web service for some functions."""

from meshed import code_to_dag, DAG
from meshed.examples import online_marketing_funcs as funcs
from meshed.tools import mk_hybrid_dag, launch_webservice, Itemgetter, AttrGetter


def test_hybrid_dag(
    dag_funcs=funcs,
    funcs_ids_to_cloudify=["cost", "revenue"],
    input_dict=dict(
        impressions=1000,
        cost_per_impression=0.02,
        click_per_impression=0.3,
        sales_per_click=0.05,
        revenue_per_sale=100,
    ),
):
    """Test hybrid dag that uses a web service for some functions.

    :param dag_funcs: list of dag functions, defaults to funcs
    :type dag_funcs: List[Callable], optional
    :param funcs_ids_to_cloudify: list of function ids to be cloudified, defaults to ['cost', 'revenue']
    :type funcs_ids_to_cloudify: list, optional
    :param input_dict: kwargs, defaults to dict( impressions=1000, cost_per_impression=0.02, click_per_impression=0.3, sales_per_click=0.05, revenue_per_sale=100 )
    :type input_dict: dict, optional
    """
    # The parameters
    dag = DAG(dag_funcs)

    print("Calling mk_hybrid_dag!")
    # Calling mk_hybrid_dag
    hybrid_dag = mk_hybrid_dag(dag, funcs_ids_to_cloudify)

    print("Starting web service!")
    with launch_webservice(hybrid_dag.funcs_to_cloudify) as ws:
        print("Web service started!")
        print("Calling dag and ws_dag!")
        dag_result = dag(**input_dict)
        print(f"dag_result: {dag_result}")
        ws_dag_result = hybrid_dag.ws_dag(**input_dict)
        print(f"ws_dag_result: {ws_dag_result}")
        assert dag_result == ws_dag_result, "Results are not equal!"
        print("Results are equal!")
        print("Done!")


def test_extractors():
    def data1():
        return {0: "start", 1: "stop", "bob": "alice"}

    first_extraction = Itemgetter([0, 1], name="first_extraction", input_name="data1")
    second_extraction = Itemgetter("bob", name="second_extraction", input_name="data1")

    from meshed import DAG

    dag1 = DAG([data1, first_extraction, second_extraction])

    dag1.synopsis_string() == """ -> data1_ -> data1
data -> first_extraction_ -> first_extraction
data -> second_extraction_ -> second_extraction"""

    assert dag1() == (("start", "stop"), "alice")

    def data2():
        from collections import namedtuple

        return namedtuple("data", ["one", "two", "bob"])("start", "stop", "alice")

    # Note: data produces a namedtuple looking like this
    d = data2()
    assert (d.one, d.two, d.bob) == ("start", "stop", "alice")

    extractor_1 = AttrGetter(["one", "two"], name="one_and_two", input_name="data2")
    extractor_2 = AttrGetter("bob", name="bob", input_name="data2")

    dag2 = DAG([data2, extractor_1, extractor_2])

    assert dag2() == (("start", "stop"), "alice")

    dag2.synopsis_string() == """ -> data2_ -> data2
data -> one_and_two_ -> one_and_two
data -> bob_ -> bob"""
