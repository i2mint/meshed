
# Copying and adding

```python
from meshed import code_to_dag

@code_to_dag
def f():
    b = f(a)
    c = g(b)
    d = h(c)
    
f.dot_digraph('rankdir="LR"')
```

<img width="662" alt="image" src="https://user-images.githubusercontent.com/1906276/167921265-eb40cc3d-91a9-474c-b4cd-2e7b417b23e9.png">


Add a copy of f to f:

```python
(f + f.copy()).dot_digraph('rankdir="LR"')
```

<img width="692" alt="image" src="https://user-images.githubusercontent.com/1906276/167921351-5167fc66-e2e5-4efb-aeb1-5aed612a676d.png">


Add a copy of f to f, but controlling the renaming function:

```python
(f + f.copy(str.upper)).dot_digraph('rankdir="LR"')
```

<img width="678" alt="image" src="https://user-images.githubusercontent.com/1906276/167921744-1f2398de-a0da-47b7-9c50-6c4d071ad266.png">


Same as above, but further rename the `A` node to be `a`, to connect the two pipelines:

```python
(f + f.copy(str.upper).copy({'A': 'a'})).dot_digraph('rankdir="LR"')
```

<img width="675" alt="image" src="https://user-images.githubusercontent.com/1906276/167921780-a1db1501-3a21-4b46-bba4-c0ee7bf3f96c.png">


# Adding edges

Let's first make a `DAG` with the nodes we want.

```python
from meshed import DAG

def f(a, b): return a + b
def g(c, d=1): return c * d
def h(x, y=1): return x ** y

three_funcs = DAG([f, g, h])

assert (
    three_funcs(x=1, c=2, a=3, b=4) 
    == (1, 2, 7) 
    == (h(x=1), g(c=2), f(a=3, b=4)) 
    == (1 ** 1, 2 * 1, 3 + 4)
)
three_funcs.dot_digraph()
```

<img width="455" alt="image" src="https://user-images.githubusercontent.com/1906276/167974316-90cfddaa-6679-4823-887c-90e997993db3.png">

See that we indeed have a `DAG` that uses all three functions, except they share no inputs, nor do they use each other's outputs. 
We can change that by added edges.

```python
hg = three_funcs.add_edge('h', 'g')
assert (
    hg(a=3, b=4, x=1)
    == (7, 1) 
    == (f(a=3, b=4), g(c=h(x=1))) 
    == (3 + 4, 1 * (1 ** 1))
)
hg.dot_digraph()
```

<img width="340" alt="image" src="https://user-images.githubusercontent.com/1906276/167974510-35b69cf6-9520-4c34-a1bd-409306d80b1a.png">


```python
fhg = three_funcs.add_edge('h', 'g').add_edge('f', 'h')
assert (
    fhg(a=3, b=4)
    == 7
    == g(h(f(3, 4)))
    == ((3 + 4) * 1) ** 1
)
fhg.dot_digraph()
```

<img width="216" alt="image" src="https://user-images.githubusercontent.com/1906276/167974558-f376b58d-f19b-4335-8e5e-67c8cbb78acf.png">


The from and to nodes can be expressed by the `FuncNode` `name` (identifier) or `out`, or even the function 
itself if it's used only once in the `DAG`.

```python
fhg = three_funcs.add_edge(h, 'g').add_edge('f_', 'h')
assert fhg(a=3, b=4) == 7
```

By default, the edge will be added from `from_node.out` to the first
parameter of the function of `to_node`.
But if you want otherwise, you can specify the parameter the edge should be
connected to.
For example, see below how we connect the outputs of `g` and `h` to the
parameters `a` and `b` of `f` respectively:

```python
f_of_g_and_h = DAG([f, g, h]).add_edge(g, f, to_param='a').add_edge(h, f, 'b')
assert (
    f_of_g_and_h(x=2, c=3, y=2, d=2)
    == 10
    == f(g(c=3, d=2), h(x=2, y=2))
    == 3 * 2 + 2 ** 2
)
f_of_g_and_h.dot_digraph()
```

<img width="285" alt="image" src="https://user-images.githubusercontent.com/1906276/167974725-895c0798-a6e5-4052-8272-518f2a3fdc33.png">


You can also add multiple edges with the `DAG.add_edges` method:

```python
fhg = DAG([f, g, h]).add_edges([(h, 'g'), ('f_', 'h')])
assert fhg(a=3, b=4) == 7
```



# Sub-DAGs

``dag[input_nodes:output_nodes]`` is the sub-dag made of intersection of all
descendants of ``input_nodes``
(inclusive) and ancestors of ``output_nodes`` (inclusive), where additionally,
when a func node is contained, it takes with it the input and output nodes
it needs.

```python
from meshed import DAG

def f(a): ...
def g(f): ...
def h(g): ...
def i(h): ...
dag = DAG([f, g, h, i])

dag.dot_digraph()
```

<img width="110" alt="image" src="https://user-images.githubusercontent.com/1906276/154749811-f9892ee6-617c-4fa6-9de9-1ebc509c04ae.png">



Get a subdag from ``g_`` (indicates the function here) to the end of ``dag``

```python
subdag = dag['g_',:]
subdag.dot_digraph()
```

<img width="100" alt="image" src="https://user-images.githubusercontent.com/1906276/154749842-c2320d1c-368d-4be8-ac57-9a77f1bb081d.png">

From the beginning to ``h_``

```python
dag[:, 'h_'].dot_digraph()
```

<img width="110" alt="image" src="https://user-images.githubusercontent.com/1906276/154750524-ece7f4b6-a3f3-46c6-a66d-7dc9b8ef254a.png">



From ``g_`` to ``h_`` (both inclusive)

```python
dag['g_', 'h_'].dot_digraph()
```

<img width="109" alt="image" src="https://user-images.githubusercontent.com/1906276/154749864-5a33aa13-0949-4aa7-945c-4d3fe7f07e7d.png">


Above we used function (node names) to specify what we wanted, but we can also
use names of input/output var-nodes. Do note the difference though.
The nodes you specify to get a sub-dag are INCLUSIVE, but when you
specify function nodes, you also get the input and output nodes of these
functions.

The ``dag['g_', 'h_']`` give us a sub-dag starting at ``f`` (the input node),
but when we ask ``dag['g', 'h_']`` instead, ``g`` being the output node of
function node ``g_``, we only get ``g -> h_ -> h``:

```python
dag['g', 'h'].dot_digraph()
```

<img width="88" alt="image" src="https://user-images.githubusercontent.com/1906276/154750753-737e2705-0ea3-4595-a93a-1567862a6edd.png">


If we wanted to include ``f`` we'd have to specify it:


```python
dag['f', 'h'].dot_digraph()
```

<img width="109" alt="image" src="https://user-images.githubusercontent.com/1906276/154749864-5a33aa13-0949-4aa7-945c-4d3fe7f07e7d.png">


Those were for simple pipelines, but let's now look at a more complex dag.

Note the definition: ``dag[input_nodes:output_nodes]`` is the sub-dag made of intersection of all 
descendants of ``input_nodes``
(inclusive) and ancestors of ``output_nodes`` (inclusive), where additionally,
when a func node is contained, it takes with it the input and output nodes
it needs.

We'll let the following examples self-comment:

```python
from meshed import DAG


def f(u, v): ...

def g(f): ...

def h(f, w): ...

def i(g, h): ...

def j(h, x): ...

def k(i): ...

def l(i, j): ...

dag = DAG([f, g, h, i, j, k, l])

dag.dot_digraph()
```

<img width="248" alt="image" src="https://user-images.githubusercontent.com/1906276/154748574-a7026125-659f-465b-9bc3-14a1864d14b2.png">

```python
dag[['u', 'f'], 'h'].dot_digraph()
```

<img width="190" alt="image" src="https://user-images.githubusercontent.com/1906276/154748685-24e706ce-b68f-429a-b7b8-7bda62ccdf36.png">


```python
dag['u', 'h'].dot_digraph()
```

<img width="183" alt="image" src="https://user-images.githubusercontent.com/1906276/154748865-6e729094-976a-4af3-87f0-b6dd3900fb8c.png">


```python
dag[['u', 'f'], ['h', 'g']].dot_digraph()
```

<img width="199" alt="image" src="https://user-images.githubusercontent.com/1906276/154748905-4eaeccbe-6cca-4492-a7a2-48f7c9937b95.png">


```python
dag[['x', 'g'], 'k'].dot_digraph()
```

<img width="133" alt="image" src="https://user-images.githubusercontent.com/1906276/154748937-7a278b25-6f0f-467c-a977-89a175e15abb.png">

```python
dag[['x', 'g'], ['l', 'k']].dot_digraph()
```

<img width="216" alt="image" src="https://user-images.githubusercontent.com/1906276/154748958-135792a6-ce16-4561-9cbe-4662113a1022.png">
