"""Coverage-pass tests for ``meshed.slabs`` and ``meshed.tools.CloudFunctions``
(no network: the HTTP client is faked)."""

import pytest

from meshed.dag import DAG
from meshed.slabs import (
    ExceptionalException,
    IteratorExit,
    Slabs,
    _get_handle_exceptions,
    _handle_exception,
    _validate_components,
    all_arguments_are_none,
    conditional_sentinel,
    do_not_break,
    log_and_return,
    output_none_if_none_arguments,
)
from meshed.tools import CloudFunctions


def _counter_slabs(n=3, **kw):
    it = iter(range(n))

    def a():
        return next(it)

    def b(a):
        return a * 10

    return Slabs(a=a, b=b, **kw)


def test_slabs_iterates_until_stopiteration():
    assert list(_counter_slabs(3)) == [
        {"a": 0, "b": 0},
        {"a": 1, "b": 10},
        {"a": 2, "b": 20},
    ]


def test_slabs_run_and_call_discard_output():
    s = _counter_slabs(2)
    assert s.run() is None
    assert _counter_slabs(2)() is None


def test_slabs_exit_value_comes_from_handler():
    s = _counter_slabs(1, handle_exceptions={StopIteration: lambda: "done"})
    list(s)
    assert s.exit_value == "done"


def test_slabs_handler_can_say_do_not_break():
    seen = []
    state = {"n": 0}

    def a():
        state["n"] += 1
        if state["n"] == 2:
            raise ValueError("skip me")
        if state["n"] > 3:
            raise StopIteration
        return state["n"]

    def handler(exc_val):
        seen.append(str(exc_val))
        return do_not_break

    s = Slabs(a=a, handle_exceptions={ValueError: handler, StopIteration: lambda: None})
    assert [x["a"] for x in s] == [1, 3]
    assert seen == ["skip me"]


def test_slabs_context_components_are_entered_and_exited():
    log = []

    class Ctx:
        def __enter__(self):
            log.append("in")
            return self

        def __exit__(self, *a):
            log.append("out")

        def __call__(self):
            raise StopIteration

    s = Slabs(ctx=Ctx())
    with s as opened:
        assert opened is s
    assert log == ["in", "out"]
    assert list(Slabs(ctx=Ctx())) == []


def test_slabs_non_callable_components_rejected():
    with pytest.raises(TypeError, match="This component is not callable: x"):
        Slabs(x=3, f=lambda: 1)
    with pytest.raises(TypeError, match="These components are not callable"):
        _validate_components({"x": 1, "y": 2})


def test_handle_exceptions_spec_normalization():
    m = _get_handle_exceptions([KeyError, ValueError])
    assert set(m) == {KeyError, ValueError}
    d = {KeyError: lambda: 1}
    assert _get_handle_exceptions(d) is d


def test_handle_exception_exact_parent_and_unmatched():
    out = _handle_exception(None, KeyError("k"), {LookupError: lambda: "parent"})
    assert out == "parent"
    out = _handle_exception(
        None, KeyError("k"), {KeyError: lambda exc_val: str(exc_val)}
    )
    assert out == "'k'"
    with pytest.raises(ExceptionalException):
        _handle_exception(None, KeyError("k"), {ValueError: lambda: 1})


def test_iterator_exit_is_base_exception_not_exception():
    assert issubclass(IteratorExit, BaseException)
    assert not issubclass(IteratorExit, Exception)


def test_log_and_return():
    logged = []
    assert log_and_return("m", logged.append) == "m"
    assert logged == ["m"]


def test_from_func_nodes_roundtrip_and_to_dag():
    def a():
        return 1

    def b(a):
        return a + 1

    dag = DAG([a, b])
    s = Slabs.from_func_nodes(dag, handle_exceptions=())
    assert list(s.components) == ["a", "b"]
    assert [fn.out for fn in s.to_func_nodes()] == ["a", "b"]
    assert isinstance(s.to_dag(), DAG)
    assert Slabs.from_dag == Slabs.from_func_nodes
    assert next(s) == {"a": 1, "b": 2}


def test_from_func_nodes_rejects_nontrivial_binds():
    from meshed.base import FuncNode

    def f(x):
        return x

    with pytest.raises(AssertionError):
        Slabs.from_func_nodes([FuncNode(f, bind={"x": "y"})])


def test_conditional_sentinel_helpers():
    assert all_arguments_are_none((None,), {"k": None})
    assert not all_arguments_are_none((None, 1), {})

    @output_none_if_none_arguments
    def add(x, y):
        return x + y

    assert add(1, 2) == 3 and add(None, y=None) is None

    guarded = conditional_sentinel(lambda a, k: not a, sentinel="empty")(len)
    assert guarded() == "empty"


# --------------------------------------------------------------------------------------
# tools.CloudFunctions


def _funcs():
    def foo(x: int, y: int = 2):
        return x + y

    def bar():
        return "bar"

    return [foo, bar]


def test_cloud_functions_mapping_interface():
    cf = CloudFunctions(_funcs(), logger=None)
    assert "foo" in cf and "nope" not in cf
    assert len(cf) == 2
    assert set(cf.keys()) == {"foo", "bar"}
    assert {k for k, _ in cf.items()} == {"foo", "bar"}
    assert len(list(cf.values())) == 2
    # a non-callable logger is silenced instead of failing
    assert cf.logger("anything") is None


def test_cloud_functions_getitem_calls_http_client_and_keeps_signature():
    import inspect

    cf = CloudFunctions(_funcs(), logger=lambda m: None)

    class FakeClient:
        def foo(self, x, y=2):
            return ("remote", x, y)

    cf.__dict__["http_client"] = FakeClient()  # prime the cached_property
    f = cf["foo"]
    assert str(inspect.signature(f)) == "(x: int, y: int = 2)"
    assert f(1, y=5) == ("remote", 1, 5)
    # "bar" is declared locally but the (fake) service doesn't have it
    with pytest.raises(KeyError):
        cf["bar"]()
    with pytest.raises(StopIteration):
        cf["unknown"]
