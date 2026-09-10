# meshed.tools

Tools to work with meshed

### Functions

| `find_funcs`(dag, func_outs)                                                                 |                                                                             |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`launch_funcs_webservice`](#meshed.tools.launch_funcs_webservice)(funcs)              | Launches a web service application with the specified functions.            |
| [`launch_webservice`](#meshed.tools.launch_webservice)(funcs_to_cloudify[, ...]) | Context manager to launch a web service application in a separate process.  |
| [`mk_dag_with_ws_funcs`](#meshed.tools.mk_dag_with_ws_funcs)(dag, ws_funcs)         | Creates a new DAG with the web service functions.                           |
| [`mk_hybrid_dag`](#meshed.tools.mk_hybrid_dag)(dag, func_ids_to_cloudify)    | Creates a hybrid DAG that uses the web service for the specified functions. |

### Classes

| `CloudFunctions`(funcs[, openapi_url, logger])   |    |
|--------------------------------------------------|----|

### meshed.tools.launch_funcs_webservice(funcs)

Launches a web service application with the specified functions.

* **Parameters:**
  **funcs** ([`list`](https://docs.python.org/3/library/stdtypes.html#list)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)]) – functions to be hosted by the web service

### meshed.tools.launch_webservice(funcs_to_cloudify, wait_after_start_seconds=10)

Context manager to launch a web service application in a separate process.

### meshed.tools.mk_dag_with_ws_funcs(dag, ws_funcs)

Creates a new DAG with the web service functions.

* **Parameters:**
  * **dag** ([`DAG`](meshed.dag.html.md#meshed.dag.DAG)) – DAG to be hybridized
  * **ws_funcs** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict)) – mapping of web service functions
* **Returns:**
  new DAG with the web service functions
* **Return type:**
  [`DAG`](meshed.dag.html.md#meshed.dag.DAG)

### meshed.tools.mk_hybrid_dag(dag, func_ids_to_cloudify)

Creates a hybrid DAG that uses the web service for the specified functions.

* **Parameters:**
  * **dag** ([`DAG`](meshed.dag.html.md#meshed.dag.DAG)) – dag to be hybridized
  * **func_ids_to_cloudify** ([`list`](https://docs.python.org/3/library/stdtypes.html#list)) – list of function ids to be cloudified
* **Returns:**
  namedtuple with funcs_to_cloudify, ws_dag and ws_funcs
* **Return type:**
  namedtuple
