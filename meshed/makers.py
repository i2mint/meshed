from numpy.random import randint, choice


def random_graph(n_nodes=7):
    """Get a random graph"""
    nodes = range(n_nodes)

    def gen():
        for src in nodes:
            n_dst = randint(0, n_nodes - 1)
            dst = choice(n_nodes, n_dst, replace=False)
            yield src, list(dst)

    return dict(gen())
