"""Test hybrid dag that uses a web service for some functions."""

from meshed import DAG
from meshed.examples import online_marketing_funcs as funcs
from meshed.tools import mk_hybrid_dag, launch_webservice


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
