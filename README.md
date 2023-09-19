# meshed

Object composition. 
In particular: Link functions up into callable objects (e.g. pipelines, DAGs, etc.)

To install: `pip install meshed`

[Documentation](https://i2mint.github.io/meshed/)

Note: The initial focuus of `meshed` was on DAGs, a versatile and probably most known kind of composition of functions, 
but `meshed` aims at capturing much more than that. 


# Quick Start

```python
from meshed import DAG

def this(a, b=1):
    return a + b
def that(x, b=1):
    return x * b
def combine(this, that):
    return (this, that)

dag = DAG((this, that, combine))
print(dag.synopsis_string())
```

    x,b -> that_ -> that
    a,b -> this_ -> this
    this,that -> combine_ -> combine


But what does it do?

It's a callable, with a signature:

```python
from inspect import signature
signature(dag)
```

    <Signature (x, a, b=1)>

And when you call it, it executes the dag from the root values you give it and
returns the leaf output values.

```python
dag(1, 2, 3)  # (a+b,x*b) == (2+3,1*3) == (5, 3)
```
    (5, 3)

```python
dag(1, 2)  # (a+b,x*b) == (2+1,1*1) == (3, 1)
```
    (3, 1)


You can see (and save image, or ascii art) the dag:

```python
dag.dot_digraph()
```

<img src="https://user-images.githubusercontent.com/1906276/127779463-ae75604b-0d69-4ac4-b206-80c2c5ae582b.png" width=200>


You can extend a dag

```python
dag2 = DAG([*dag, lambda this, a: this + a])
dag2.dot_digraph()
```

<img src="https://user-images.githubusercontent.com/1906276/127779748-70b47907-e51f-4e64-bc18-9545ee07e632.png" width=200>

You can get a sub-dag by specifying desired input(s) and outputs.

```python
dag2[['that', 'this'], 'combine'].dot_digraph()
```

<img src="https://user-images.githubusercontent.com/1906276/127779781-8aac40eb-ed52-4694-b50e-4af896cc30a2.png" width=150>



## Note on flexibility

The above DAG was created straight from the functions, using only the names of the
functions and their parameters to define how to hook the network up.

But if you didn't write those functions specifically for that purpose, or you want
to use someone else's functions, one would need to specify the relation between parameters, inputs and outputs.

For that purpose, functions can be adapted using the class FuncNode. The class allows you to essentially rename each of the parameters and also specify which output should be used as an argument for any other functions.

Let us consider the example below.

```python
def f(a, b):
    return a + b

def g(a_plus_b, d):
    return a_plus_b * d
```

Say we want the output of f to become the value of the parameter a_plus_b. We can do that by assigning the string 'a_plus_b' to the out parameter of a FuncNode representing the function f:

```python
f_node = FuncNode(func=f, out="a_plus_b")
```

We can now create a dag using our f_node instead of f:

```python
dag = DAG((f_node, g))
```

Our dag behaves as wanted:

```python
dag(a=1, b=2, d=3)
9
```

Now say we would also like for the value given to b to be also given to d. We can achieve that by binding d to b in the bind parameter of a FuncNode representing g:

```python
g_node = FuncNode(func=g, bind={"d": "b"})
```

The dag created with f_node and g_node has only two parameters, namely a and b:

```python
dag = DAG((f_node, g_node))
dag(a=1, b=2)
6
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



# Examples

## A train/test ML pipeline

Consider a simple train/test ML pipeline that looks like this.

![image](https://user-images.githubusercontent.com/1906276/135151068-179d958e-9e96-48aa-9188-52ae22919c6e.png)

With this, we might decide we want to give the user control over how to do 
`train_test_split` and `learner`, so we offer this interface to the user:

![image](https://user-images.githubusercontent.com/1906276/135151094-661850c0-f10c-49d8-ace2-46b3d994de80.png)

With that, the user can just bring its own `train_test_split` and `learner` 
functions, and as long as it satisfied the 
expected (and even better; declared and validatable) protocol, things will work. 

In some situations we'd like to fix some of how `train_test_split` and 
`learner` work, allowing the user to control only some aspects of them. 
This function would look like this:

![image](https://user-images.githubusercontent.com/1906276/135151137-3d9a290f-d5e7-4f24-a418-82f1edb8a46a.png)

And inside, it does:

![image](https://user-images.githubusercontent.com/1906276/135151114-926b52b8-0536-4565-bd56-95099f21e4ff.png)

`meshed` allows us to easily manipulate such functional structures to 
adapt them to our needs.


# itools module
Tools that enable operations on graphs where graphs are represented by an adjacency Mapping.

Again. 

Graphs: You know them. Networks. 
Nodes and edges, and the ecosystem descriptive or transformative functions surrounding these.
Few languages have builtin support for the graph data structure, but all have their libraries to compensate.

The one you're looking at focuses on the representation of a graph as `Mapping` encoding 
its [adjacency list](https://en.wikipedia.org/wiki/Adjacency_list). 
That is, a dictionary-like interface that specifies the graph by specifying for each node
what nodes it's adjacent to:

```python
assert graph[source_node] == set_of_nodes_that_source_node_has_edges_to
```

We emphasize that there is no specific graph instance that you need to squeeze your graph into to
be able to use the functions of `meshed`. Suffices that your graph's structure is expressed by 
that dict-like interface 
-- which grown-ups call `Mapping` (see the `collections.abc` or `typing` standard libs for more information).

You'll find a lot of `Mapping`s around pythons. 
And if the object you want to work with doesn't have that interface, 
you can easily create one using one of the many tools of `py2store` meant exactly for that purpose.


# Examples

```pydocstring
>>> from meshed.itools import edges, nodes, isolated_nodes
>>> graph = dict(a='c', b='ce', c='abde', d='c', e=['c', 'b'], f={})
>>> sorted(edges(graph))
[('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'), ('d', 'c'), ('e', 'b'), ('e', 'c')]
>>> sorted(nodes(graph))
['a', 'b', 'c', 'd', 'e', 'f']
>>> set(isolated_nodes(graph))
{'f'}
>>>
>>> from meshed.makers import edge_reversed_graph
>>> g = dict(a='c', b='cd', c='abd', e='')
>>> assert edge_reversed_graph(g) == {'c': ['a', 'b'], 'd': ['b', 'c'], 'a': ['c'], 'b': ['c'], 'e': []}
>>> reverse_g_with_sets = edge_reversed_graph(g, set, set.add)
>>> assert reverse_g_with_sets == {'c': {'a', 'b'}, 'd': {'b', 'c'}, 'a': {'c'}, 'b': {'c'}, 'e': set([])}
```
