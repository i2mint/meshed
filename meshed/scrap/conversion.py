"""Utils to convert graphs from one specification to another"""

import os

DFLT_PROG = "neato"
graph_template = 'strict graph "" {{\n{dot_str}\n}}'
digraph_template = 'strict digraph "" {{\n{dot_str}\n}}'


def ensure_dot_code(x: str):
    if not x.startswith("strict"):
        if "--" in x:
            print("asdfdf")
            x = graph_template.format(dot_str=x)
        else:  # if '->' in dot_str
            x = digraph_template.format(dot_str=x)
    return x


def dot_to_nx(dot_src):
    from pygraphviz import AGraph

    dot_src = ensure_dot_code(dot_src)
    return AGraph(string=dot_src)


def dot_to_ipython_image(dot_src, *, prog=DFLT_PROG, tmp_file="__tmp_file.png"):
    from IPython.display import Image

    dot_src = ensure_dot_code(dot_src)
    g = dot_to_nx(dot_src)
    g.draw(tmp_file, prog=prog)
    ipython_obj = Image(tmp_file)

    if os.path.isfile(tmp_file):
        os.remove(tmp_file)

    return ipython_obj


def dot_to_pydot(dot_src):
    import pydot

    dot_src = ensure_dot_code(dot_src)
    return pydot.graph_from_dot_data(dot_src)
