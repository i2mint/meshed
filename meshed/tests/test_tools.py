import i2

from meshed import code_to_dag, DAG
from meshed.examples import online_marketing_funcs as funcs
from meshed.tools import mk_hybrid_dag, launch_webservice

assert str(i2.Sig(mk_hybrid_dag)) == '(dag, func_ids_to_cloudify)'

mk_hybrid_dag.dot_digraph()

# The parameters
dag = DAG(funcs)
funcs_ids_to_cloudify = ['cost', 'revenue']
input_dict = dict(
    impressions=1000,
    cost_per_impression=0.02,
    click_per_impression=0.3,
    sales_per_click=0.05,
    revenue_per_sale=100
)

print("Calling mk_hybrid_dag!")
# Calling mk_hybrid_dag
ws_dag = mk_hybrid_dag(dag, funcs_ids_to_cloudify)
print(type(ws_dag))
print(ws_dag)

print("Starting web service!")
with launch_webservice(ws_dag.ws_app):
    dag_result = dag(**input_dict)
    ws_dag_result = ws_dag(**input_dict)
    assert dag_result == ws_dag_result
