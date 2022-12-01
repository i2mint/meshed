from meshed.makers import code_to_dag


@code_to_dag
def dag_01():
    b = f(a)
    c = g(a)
    d = h(b, c)


@code_to_dag
def dag_02():
    b = f(a)
    c = g(x=a)
    d = h(y=b, c=c)


_string01 = '''a -> f -> b
a -> g -> c
b,c -> h -> d'''

_string02 = '''a -> f -> b
x=a -> g -> c
y=b,c -> h -> d'''


def test_synopsis_string():
    s11 = '\n'.join([fn.synopsis_string() for fn in dag_01.func_nodes])
    s21 = '\n'.join(
        [fn.synopsis_string(bind_info='hybrid') for fn in dag_01.func_nodes]
    )
    assert s11 == _string01 == s21
    s22 = '\n'.join(
        [fn.synopsis_string(bind_info='hybrid') for fn in dag_02.func_nodes]
    )
    assert s22 == _string02
