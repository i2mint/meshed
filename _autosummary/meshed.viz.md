# meshed.viz

Visualization utilities for the meshed package.

### Functions

| [`add_new_line_if_none`](#meshed.viz.add_new_line_if_none)(s)                           | Since graphviz 0.18, need to have a newline in body lines.   |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`dot_lines_of_func_nodes`](#meshed.viz.dot_lines_of_func_nodes)(objs[, start_lines, ...]) | Get lines generator for the graphviz.DiGraph(body=list(...)) |
| [`dot_lines_of_objs`](#meshed.viz.dot_lines_of_objs)(objs[, start_lines, end_lines]) | Get lines generator for the graphviz.DiGraph(body=list(...)) |
| `visualize_graph`(graph)                                                                           |                                                              |
| `visualize_graph_interactive`(graph)                                                               |                                                              |

### meshed.viz.add_new_line_if_none(s)

Since graphviz 0.18, need to have a newline in body lines.
This util is there to address that, adding newlines to body lines
when missing.

### meshed.viz.dot_lines_of_func_nodes(objs, start_lines=(), end_lines=(), \*\*kwargs)

Get lines generator for the graphviz.DiGraph(body=list(…))

```pycon
>>> from meshed.base import FuncNode
>>> def add(a, b=1):
...     return a + b
>>> def mult(x, y=3):
...     return x * y
>>> def exp(mult, a):
...     return mult ** a
>>> func_nodes = [
...     FuncNode(add, out='x'),
...     FuncNode(mult, name='the_product'),
...     FuncNode(exp)
... ]
>>> lines = list(dot_lines_of_objs(func_nodes))
>>> assert lines == [
... 'x [label="x" shape="none"]',
... '_add [label="_add" shape="box"]',
... '_add -> x',
... 'a [label="a" shape="none"]',
... 'b [label="b=" shape="none"]',
... 'a -> _add',
... 'b -> _add',
... 'mult [label="mult" shape="none"]',
... 'the_product [label="the_product" shape="box"]',
... 'the_product -> mult',
... 'x [label="x" shape="none"]',
... 'y [label="y=" shape="none"]',
... 'x -> the_product',
... 'y -> the_product',
... 'exp [label="exp" shape="none"]',
... '_exp [label="_exp" shape="box"]',
... '_exp -> exp',
... 'mult [label="mult" shape="none"]',
... 'a [label="a" shape="none"]',
... 'mult -> _exp',
... 'a -> _exp'
... ]
```

```pycon
>>> from meshed.util import dot_to_ascii
>>>
>>> print(dot_to_ascii('\n'.join(lines)))

                a        ─┐
                          │
           │              │
           │              │
           ▼              │
         ┌─────────────┐  │
 b=  ──▶ │    _add     │  │
         └─────────────┘  │
           │              │
           │              │
           ▼              │
                          │
                x         │
                          │
           │              │
           │              │
           ▼              │
         ┌─────────────┐  │
 y=  ──▶ │ the_product │  │
         └─────────────┘  │
           │              │
           │              │
           ▼              │
                          │
              mult        │
                          │
           │              │
           │              │
           ▼              │
         ┌─────────────┐  │
         │    _exp     │ ◀┘
         └─────────────┘
           │
           │
           ▼

               exp
```

### meshed.viz.dot_lines_of_objs(objs, start_lines=(), end_lines=(), \*\*kwargs)

Get lines generator for the graphviz.DiGraph(body=list(…))

```pycon
>>> from meshed.base import FuncNode
>>> def add(a, b=1):
...     return a + b
>>> def mult(x, y=3):
...     return x * y
>>> def exp(mult, a):
...     return mult ** a
>>> func_nodes = [
...     FuncNode(add, out='x'),
...     FuncNode(mult, name='the_product'),
...     FuncNode(exp)
... ]
>>> lines = list(dot_lines_of_objs(func_nodes))
>>> assert lines == [
... 'x [label="x" shape="none"]',
... '_add [label="_add" shape="box"]',
... '_add -> x',
... 'a [label="a" shape="none"]',
... 'b [label="b=" shape="none"]',
... 'a -> _add',
... 'b -> _add',
... 'mult [label="mult" shape="none"]',
... 'the_product [label="the_product" shape="box"]',
... 'the_product -> mult',
... 'x [label="x" shape="none"]',
... 'y [label="y=" shape="none"]',
... 'x -> the_product',
... 'y -> the_product',
... 'exp [label="exp" shape="none"]',
... '_exp [label="_exp" shape="box"]',
... '_exp -> exp',
... 'mult [label="mult" shape="none"]',
... 'a [label="a" shape="none"]',
... 'mult -> _exp',
... 'a -> _exp'
... ]
```

```pycon
>>> from meshed.util import dot_to_ascii
>>>
>>> print(dot_to_ascii('\n'.join(lines)))

                a        ─┐
                          │
           │              │
           │              │
           ▼              │
         ┌─────────────┐  │
 b=  ──▶ │    _add     │  │
         └─────────────┘  │
           │              │
           │              │
           ▼              │
                          │
                x         │
                          │
           │              │
           │              │
           ▼              │
         ┌─────────────┐  │
 y=  ──▶ │ the_product │  │
         └─────────────┘  │
           │              │
           │              │
           ▼              │
                          │
              mult        │
                          │
           │              │
           │              │
           ▼              │
         ┌─────────────┐  │
         │    _exp     │ ◀┘
         └─────────────┘
           │
           │
           ▼

               exp
```
