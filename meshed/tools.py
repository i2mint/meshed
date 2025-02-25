"""Tools to work with meshed"""

from collections import namedtuple
from contextlib import contextmanager
from functools import cached_property, partial
import multiprocessing
import os
import time
from typing import Callable, List
from urllib.parse import urljoin

import i2

from meshed.dag import DAG


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 3030))
API_URL = os.environ.get("API_URL", f"http://localhost:{PORT}")
SERVER = os.environ.get("SERVER", "wsgiref")
OPENAPI_URL = urljoin(API_URL, "openapi")


def find_funcs(dag, func_outs):
    return list(dag.find_funcs(lambda x: x.out in func_outs))


def mk_dag_with_ws_funcs(dag: DAG, ws_funcs: dict) -> DAG:
    """Creates a new DAG with the web service functions.

    :param dag: DAG to be hybridized
    :type dag: DAG
    :param ws_funcs: mapping of web service functions
    :type ws_funcs: dict
    :return: new DAG with the web service functions
    :rtype: DAG
    """
    return dag.ch_funcs(**ws_funcs)


def launch_funcs_webservice(funcs: List[Callable]):
    """Launches a web service application with the specified functions.

    :param funcs: functions to be hosted by the web service
    :type funcs: List[Callable]
    """
    from extrude import mk_api, run_api

    ws_app = mk_api(funcs, openapi=dict(base_url=API_URL))
    run_api(ws_app, host=HOST, port=PORT, server=SERVER)


@contextmanager
def launch_webservice(funcs_to_cloudify, wait_after_start_seconds=10):
    """Context manager to launch a web service application in a separate process."""
    ws = multiprocessing.Process(
        target=launch_funcs_webservice, args=(funcs_to_cloudify,)
    )
    ws.start()
    # TODO: I prefer using a timeout instead of a fixed wait time
    # TODO: Use strand tool for this: https://github.com/i2mint/strand/blob/7443631e9d2486358f0a34ed182e85b6ded5e50c/strand/taskrunning/utils.py#L54
    time.sleep(wait_after_start_seconds)
    yield ws

    ws.terminate()


class CloudFunctions:
    def __init__(self, funcs: List[Callable], openapi_url=OPENAPI_URL, logger=print):
        """Creates a Python dictionary-like object that maps the web service functions to Python functions.

        :param funcs: list of functions hosted by the web service
        :type funcs: List[Callable]
        :param openapi_url: url to get openapi spec, defaults to "http://localhost:{PORT}/openapi"
        :type openapi_url: str, optional
        :param logger: logger function, defaults to print
        :type logger: Callable, optional
        """
        self.funcs = funcs
        self.openapi_url = openapi_url
        self.logger = logger if callable(logger) else lambda x: None

    @cached_property
    def http_client(self):
        from http2py import HttpClient

        try:
            return HttpClient(url=self.openapi_url)
        except Exception:
            self.logger(
                f"Could not connect to {self.openapi_url}. Waiting 10 seconds and trying again."
            )
            time.sleep(10)
            return HttpClient(url=self.openapi_url)

    @cached_property
    def func_names(self):
        return frozenset(f.__name__ for f in self.funcs)

    def __getitem__(self, key):
        """Returns a Python function that calls the web service function.
        HttpClient is queried at the execution of the function.
        """

        @i2.Sig(next(f for f in self.funcs if key == f.__name__))
        def ws_func(*a, **kw):
            self.logger(f"Getting web service for: {key}")
            if (_wsf := getattr(self.http_client, key, None)) is not None:
                self.logger(f"Found web service for: {key}")
                return _wsf(*a, **kw)
            raise KeyError(key)

        return ws_func

    def __contains__(self, key):
        return key in self.func_names

    def __len__(self):
        return len(self.func_names)

    def keys(self):
        return self.func_names

    def values(self):
        return (self[k] for k in self.keys())

    def items(self):
        return ((k, self[k]) for k in self.keys())


def mk_hybrid_dag(dag: DAG, func_ids_to_cloudify: list):
    """Creates a hybrid DAG that uses the web service for the specified functions.

    :param dag: dag to be hybridized
    :type dag: DAG
    :param func_ids_to_cloudify: list of function ids to be cloudified
    :type func_ids_to_cloudify: list
    :return: namedtuple with funcs_to_cloudify, ws_dag and ws_funcs
    :rtype: namedtuple
    """
    funcs_to_cloudify = find_funcs(dag, func_ids_to_cloudify)
    ws_funcs = CloudFunctions(funcs_to_cloudify)
    ws_dag = mk_dag_with_ws_funcs(dag, ws_funcs)

    HybridDAG = namedtuple("HybridDAG", ["funcs_to_cloudify", "ws_dag", "ws_funcs"])
    return HybridDAG(funcs_to_cloudify, ws_dag, ws_funcs)
