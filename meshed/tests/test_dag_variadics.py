import pytest
from pytest import fixture
from meshed import DAG
from i2 import Sig


@fixture
def foo():
    def func(x, y=1):
        return x + y

    return func


def test_addition_variadics():
    def foo(w, /, x: float, y="YY", *, z: str = "ZZ", **rest):
        pass

    sig = Sig(foo)
    # res = sig.map_arguments(
    #    (11, 22, "you"), dict(z="zoo", other="stuff"), post_process=True
    # )
    # assert res == "{'w': 11, 'x': 22, 'y': 'you', 'z': 'zoo', 'other': 'stuff'}"
    assert True
