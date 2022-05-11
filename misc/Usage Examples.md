
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
