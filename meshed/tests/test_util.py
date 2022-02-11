import meshed as ms


def test_name_of_obj():
    assert ms.util.name_of_obj(map) == 'map'
    assert ms.util.name_of_obj([1, 2, 3])
    'list'
    assert ms.util.name_of_obj(print) == 'print'
    assert ms.util.name_of_obj(lambda x: x) == '<lambda>'
    from functools import partial

    assert ms.util.name_of_obj(partial(print, sep=',')) == 'print'


def test_incremental_str_maker():
    lambda_name = ms.util.incremental_str_maker(str_format='lambda_{:03.0f}')
    assert lambda_name() == 'lambda_001'
    assert lambda_name() == 'lambda_002'


def test_func_name():
    def my_func(a=1, b=2):
        return a * b

    assert ms.util.func_name(my_func) == 'my_func'
    assert ms.util.func_name(lambda x: x).startswith('lambda_')
