from typing import Mapping
from collections import ChainMap
from meshed.itools import edge_reversed_graph
from i2 import Sig


def get_first_item_and_assert_unicity(seq):
    seq_length = len(seq)
    if seq_length:
        assert seq_length == 1, (
            f'There should be one and one only item in the ' f'sequence: {seq}'
        )
        return seq[0]
    else:
        return None


def func_node_names_and_outs(dag):
    for func_node in dag.func_nodes:
        yield func_node.name, func_node.out


NoSuchKey = type('NoSuchKey', (), {})

# TODO: Cache validation and invalidation
class CachedDag:
    def __init__(self, dag, cache=True, name=None):
        self.dag = dag
        self.reversed_graph = edge_reversed_graph(dag.graph_ids)
        self.roots = set(self.dag.roots)
        self.leafs = set(self.dag.leafs)
        self.var_nodes = set(self.dag.var_nodes)
        self.func_node_of_id = {fn.out: fn for fn in self.dag.func_nodes}
        self.name = name
        self.out_of_func_node_name = dict(func_node_names_and_outs(self.dag))
        self.defaults = Sig(self.dag).defaults
        if cache is True:
            self.cache = {}
        elif not isinstance(cache, Mapping):
            raise NotImplementedError(
                'This type of cache is not implemented (must resolve to a Mapping): '
                f'{cache=}'
            )
        self._cache = ChainMap(self.defaults, self.cache)

    @property
    def __name__(self):
        return self.name or self.dag.__name__

    def __iter__(self):
        yield from self.reversed_graph

    def func_node_id(self, k):
        func_node_name = get_first_item_and_assert_unicity(self.reversed_graph[k])
        if func_node_name is not None:
            return self.out_of_func_node_name[func_node_name]

    def __call__(self, k, input_kwargs=()):
        #         print(f"Calling ({k=},{input_kwargs=})\t{self.cache=}")
        if intersection := (input_kwargs.keys() & self.cache.keys()):
            raise ValueError(
                f"input_kwargs can't contain any keys that are already in cache! "
                f'These names were in both: {intersection}'
            )
        _cache = ChainMap(input_kwargs, self._cache)
        if k in _cache:
            return _cache[k]
        input_kwargs = dict(input_kwargs)
        func_node_id = self.func_node_id(k)
        #         print(f"{func_node_id=}")
        if func_node_id:
            if (output := self.cache.get(func_node_id)) is not None:
                return output
            else:
                func_node = self.func_node_of_id[func_node_id]
                input_sources = {
                    src: self(src, input_kwargs) for src in func_node.bind.values()
                }
                #                 inputs = dict(input_sources, **input_kwargs)  #
                #                 TODO: do we need to include **self.defaults in the
                #                  middle?
                inputs = ChainMap(_cache, input_sources)
                #                 print(f"Computing {func_node_id}: ", end=" ")
                output = func_node.call_on_scope(inputs, write_output_into_scope=False)
                self.cache[func_node_id] = output
                #                 print(f"result -> {output}")
                return output
        else:  # k is a root node
            inputs = ChainMap(input_kwargs, self._cache)
            assert k in self.roots, f'Was expecting this to be a root node: {k}'
            if (output := inputs.get(k, NoSuchKey)) is not NoSuchKey:
                return output
            else:
                raise TypeError(
                    f'The input_kwargs of a {self.__name__} call is missing 1 required '
                    f"argument: '{k}'"
                )


def cached_dag_test():
    def add(a, b=1):
        return a + b

    def mult(x, y=2):
        return x * y

    def exp(mult, n=3):
        return m ** n

    def subtract(a, b=4):
        return a - b

    from meshed import code_to_dag

    @code_to_dag(func_src=locals())
    def dag(w, ww, www):
        x = mult(w, ww)
        y = add(x, www)
        z = subtract(x, y)

    g = CachedDag(dag)

    assert g('z', {'w': 2, 'ww': 3, 'www': 4}) == -4 == dag(2, 3, 4)
