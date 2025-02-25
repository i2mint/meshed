"""Wrapping dags"""

from meshed import DAG


class DDag(DAG):
    wrappers = ()

    def _call(self, *args, **kwargs):
        if not self.wrappers:
            return super()._call(*args, **kwargs)
        else:
            decorator = Line(*self.wrappers)
            decorated_dag_call = decorator(super()._call)
            return decorated_dag_call(*args, **kwargs)


def test_ddag():
    def f(a, b=2):
        return a + b

    def g(f, c=3):
        return f * c

    # d = DDag([f, g])
    d = DDag([f, g])

    d.dot_digraph()

    assert d(1, 2, 3) == 9  # can call
    from i2 import Sig

    assert str(Sig(d)) == "(a, b=2, c=3)"  # has correct signature

    def dec(func):
        def _dec(*args, **kwargs):
            print(func.__name__, args, kwargs)
            return func(*args, **kwargs)

        return _dec

    def rev(func):
        def _rev(*args, **kwargs):
            assert not kwargs, "Can't have keyword arguments with rev"
            return func(*args[::-1])

        return _rev

    d.wrappers = (dec,)

    assert d(1, 2, 3) == 9
    # prints: _call (1, 2, 3) {}
    assert d(1, 2, c=3) == 9
    # prints: _call (1, 2) {'c': 3}

    d.wrappers = (dec, rev)
    assert d(1, 2, 3) == 5
    # prints: _call (3, 2, 1) {}
