import meshed as ms
import pytest

import meshed.base
import meshed.util


@pytest.fixture
def example_func_nodes():
    def f(a=1, b=2):
        return 12

    def g(c=3):
        return 42

    func_nodes = [f, g]
    result = meshed.base._mk_func_nodes(func_nodes)
    return result


def test_find_first_free_name():
    prefix = "ab"
    exclude_names = ("cd", "lm", "ab", "ab__0", "ef")
    assert (
        meshed.util.find_first_free_name(
            prefix, exclude_names=exclude_names, start_at=0
        )
        == "ab__1"
    )


def test_mk_func_name():
    def myfunc1(a=1, b=3, c=1):
        return a + b * c

    assert meshed.util.mk_func_name(myfunc1, exclude_names=("myfunc1")) == "myfunc1__2"


def test_arg_names():
    def myfunc1(a=1, b=3, c=1):
        return a + b * c

    args_list = meshed.util.arg_names(myfunc1, "myfunc1", exclude_names=("a", "b"))
    assert args_list == ["myfunc1__a", "myfunc1__b", "c"]


def test_named_partial():
    f = meshed.util.named_partial(print, sep="\\n")
    assert f.__name__ == "print"
    g = meshed.util.named_partial(print, sep="\\n", __name__="now_partial_has_a_name")
    assert g.__name__ == "now_partial_has_a_name"


def test_hook_up():
    def formula1(w, /, x: float, y=1, *, z: int = 1):
        return ((w + x) * y) ** z

    d = {}
    f = ms.dag.hook_up(formula1, d)
    d.update(w=2, x=3, y=4)
    f()
    assert d == {"w": 2, "x": 3, "y": 4, "formula1": 20}
    d.clear()
    d.update(w=1, x=2, y=3)
    f()
    assert d["formula1"] == 9


def test_complete_dict_with_iterable_of_required_keys():
    d = {"a": "A", "c": "C"}
    meshed.base._complete_dict_with_iterable_of_required_keys(d, "abc")
    assert d == {"a": "A", "c": "C", "b": "b"}


def test_inverse_dict_asserting_losslessness():
    d = {"w": 2, "x": 3, "y": 4, "formula1": 20}
    d_inv = meshed.util.inverse_dict_asserting_losslessness(d)
    assert d_inv == {2: "w", 3: "x", 4: "y", 20: "formula1"}


def test_mapped_extraction():
    extracted = meshed.base._mapped_extraction(
        src={"A": 1, "B": 2, "C": 3}, to_extract={"a": "A", "c": "C", "d": "D"}
    )
    assert dict(extracted) == {"a": 1, "c": 3}


def test_underscore_func_node_names_maker():
    def func_1():
        pass

    name, out = meshed.base.underscore_func_node_names_maker(
        func_1, name="init_func", out="output_name"
    )
    assert name, out == ("init_func", "output_name")
    assert meshed.base.underscore_func_node_names_maker(func_1) == (
        "func_1_",
        "func_1",
    )
    assert meshed.base.underscore_func_node_names_maker(func_1, name="init_func") == (
        "init_func",
        "_init_func",
    )
    assert meshed.base.underscore_func_node_names_maker(func_1, out="output_name") == (
        "func_1",
        "output_name",
    )


def test_duplicates():
    assert meshed.base.duplicates("abbaaeccf") == ["a", "b", "c"]


def test_FuncNode():
    def multiply(x, y):
        return x * y

    item_price = 3.5
    num_of_items = 2
    func_node = meshed.base.FuncNode(
        func=multiply,
        bind={"x": "item_price", "y": "num_of_items"},
    )
    assert (
        str(func_node) == "FuncNode(x=item_price,y=num_of_items -> multiply_ -> "
        "multiply)"
    )
    scope = {"item_price": 3.5, "num_of_items": 2}
    assert func_node.call_on_scope(scope) == 7.0
    assert scope == {"item_price": 3.5, "num_of_items": 2, "multiply": 7.0}
    # Give a name to output
    assert (
        str(
            meshed.base.FuncNode(
                func=multiply,
                name="total_price",
                bind={"x": "item_price", "y": "num_of_items"},
            )
        )
        == "FuncNode(x=item_price,y=num_of_items -> total_price -> _total_price)"
    )
    # rename the function and the output
    assert (
        str(
            meshed.base.FuncNode(
                func=multiply,
                name="total_price",
                bind={"x": "item_price", "y": "num_of_items"},
                out="daily_expense",
            )
        )
        == "FuncNode(x=item_price,y=num_of_items -> total_price -> daily_expense)"
    )


def test_mk_func_nodes():
    def f(a=1, b=2):
        return 12

    def g(c=3):
        return 42

    func_nodes = [f, g]
    result = meshed.base._mk_func_nodes(func_nodes)
    assert str(list(result)) == "[FuncNode(a,b -> f_ -> f), FuncNode(c -> g_ -> g)]"


def test_func_nodes_to_graph_dict(example_func_nodes):
    fnodes = example_func_nodes
    result = meshed.base._func_nodes_to_graph_dict(fnodes)
    assert True
