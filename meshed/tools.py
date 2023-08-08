"""Tools to work with meshed"""

from collections import namedtuple
from contextlib import contextmanager
from functools import cached_property, partial
import multiprocessing
import os
import time

import i2


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 3030))
API_URL = os.environ.get("API_URL", f"http://localhost:{PORT}")
SERVER = os.environ.get("SERVER", "wsgiref")

def find_funcs(dag, func_outs):
    return list(dag.find_funcs(lambda x: x.out in func_outs))

def mk_dag_with_wf_funcs(dag, ws_funcs):
    return dag.ch_funcs(**ws_funcs)

def launch_funcs_webservice(funcs):
    from extrude import mk_api, run_api

    ws_app = mk_api(funcs, openapi=dict(base_url=API_URL))
    run_api(ws_app, host=HOST, port=PORT, server=SERVER)

@contextmanager
def launch_webservice(funcs_to_cloudify):
    """Launches a web service application in a separate process."""

    ws = multiprocessing.Process(target=launch_funcs_webservice, args=(funcs_to_cloudify,))
    ws.start()
    time.sleep(5)
    yield ws

    ws.terminate()


class PyBinderFuncs:
    def __init__(self, ws_app, funcs):
        self.ws_app = ws_app
        self.funcs = funcs

    @cached_property
    def http_client(self):
        from urllib.parse import urljoin
        from http2py import HttpClient

        return HttpClient(url=urljoin(API_URL, 'openapi'))

    @cached_property
    def func_names(self):
        return frozenset(f.__name__ for f in self.funcs)

    def __getitem__(self, key):
        """Returns a Python function that calls the web service function.
        HttpClient is queried at the execution of the function.
        """
        @i2.Sig(next(f for f in self.funcs if key == f.__name__))
        def ws_func(*a, **kw):
            print(f'Getting web service for: {key}')
            if (_wsf := getattr(self.http_client, key, None)) is not None:
                print(f'Found web service for: {key}')
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


def py_binder_funcs(ws_app, funcs):
    """Maps the web service application to Python functions."""

    return PyBinderFuncs(ws_app, funcs)

def mk_web_service(funcs):
    """Makes a web service application from a list of functions."""
    from py2http import mk_app

    return mk_app(funcs, openapi=dict(base_url=API_URL))

def mk_hybrid_dag(dag, func_ids_to_cloudify):
    funcs_to_cloudify = find_funcs(dag, func_ids_to_cloudify)
    ws_app = mk_web_service(funcs_to_cloudify)
    ws_funcs = py_binder_funcs(ws_app, funcs_to_cloudify)
    ws_dag = mk_dag_with_wf_funcs(dag, ws_funcs)

    HybridDAG = namedtuple('HybridDAG', ['funcs_to_cloudify', 'ws_app', 'ws_dag', 'ws_funcs'])
    return HybridDAG(funcs_to_cloudify, ws_app, ws_dag, ws_funcs)
