# meshed: Design Document

**A Declarative Dataflow Composition Framework for Python**

---

## 1. Motivation and Problem Statement

### The Composition Problem

Software systems are built from compositions of smaller operations. Python's standard tools for function composition are surprisingly impoverished:

- **Direct function calls** (`f(g(x))`) work but create tightly coupled, deeply nested expressions that conflate dataflow topology with execution order.
- **Linear pipelines** (e.g., `lined.Line`, Unix pipes, `functools.reduce`) sequence operations but cannot express fan-out (one output feeding multiple consumers), fan-in (multiple inputs converging), or diamond dependencies.
- **Manual wiring** (glue code) is the predominant approach -- developers write procedural code that sources arguments, calls functions, and routes outputs. This "plumbing" code obscures the actual dataflow structure.

Consider this concrete scenario: you have functions `f(a, b) -> c`, `g(c, d) -> e`, and `h(c, e) -> result`. The dataflow has a diamond dependency on `c`. In raw Python:

```python
c = f(a, b)
e = g(c, d)
result = h(c, e)
```

This procedural code buries the dataflow topology inside execution order. You cannot:
- Inspect the dependency graph programmatically
- Substitute `f` for a different implementation without modifying the calling code
- Re-render the same computation as a REST API, a CLI, or a reactive UI
- Serialize the computation structure independently of the function implementations

### From Pipelines to DAGs to Meshes

The progression of composition expressiveness follows a clear hierarchy:

1. **Pipelines** (linear composition): `a -> f -> b -> g -> c`. One input, one output per step. Adequate for sequential transformations but cannot express concurrency, branching, or merging.

2. **DAGs** (directed acyclic graphs): Allow fan-in, fan-out, and diamond dependencies. The `meshed.DAG` captures this level. Adequate for batch computation workflows where all inputs are available upfront.

3. **Meshes** (the full vision): Heterogeneous node types (functions, storage interfaces, UI event sources, stream processors), potentially cyclic relationships (reactive propagation), multiple execution strategies, and polymorphic sourcing. This is what `meshed` aspires to but has only partially implemented.

The key insight driving `meshed` is that **the topology of dataflow is a first-class specification** that should be decoupled from execution semantics. A DAG should be an inspectable, transformable, serializable object -- not just a sequence of function calls.

### Scope vs. Existing Solutions

`meshed` occupies a specific niche distinct from workflow orchestrators (Airflow, Prefect, Dagster) and distributed compute frameworks (Dask, Ray):

- **Workflow orchestrators** focus on task scheduling, retries, monitoring, and distributed execution of coarse-grained steps. They are heavy-weight and infrastructure-oriented.
- **Distributed compute frameworks** focus on parallelism and data distribution across clusters.
- **meshed** focuses on **in-process function composition** with emphasis on introspection, graph manipulation (sub-DAG extraction, partial application, function substitution), and progressive disclosure. It targets the scale of a single application's internal wiring, not cluster orchestration.

The closest relatives are `dask.delayed` (lazy graph construction), `graphlib` (topological sorting only), and `ploomber` (pipeline construction). `meshed` distinguishes itself through convention-based implicit wiring, rich graph manipulation APIs (`__getitem__` slicing, `partial`, `ch_funcs`, `ch_names`), and the ambition to render the same graph to multiple execution backends [1].


## 2. Core Abstractions and Data Model

### 2.1 FuncNode: The Computation Vertex

`FuncNode` [2] is a `@dataclass` wrapping a callable with explicit specification of its dataflow interface:

| Field | Type | Purpose |
|-------|------|---------|
| `func` | `Callable` | The wrapped computation |
| `name` | `str` | The identifier of this computation in the graph (the "function node" identity) |
| `bind` | `dict[str, str]` | `{param_name: var_node_name}` -- maps function parameters to variable nodes |
| `out` | `str` | The variable node where the function writes its output |

The critical method is `call_on_scope(scope: MutableMapping)` [2:L327], which:
1. Extracts input values from `scope` using `bind` as a renaming map
2. Calls `func` with those arguments (via `call_somewhat_forgivingly`)
3. Writes the result to `scope[self.out]`

This design realizes the **scope-based execution model**: a FuncNode reads from and writes to a shared dictionary, making it a reified assignment statement. As the docstring explains: `total_price = multiply(item_price, num_of_items)` is equivalent to a FuncNode that reads `item_price` and `num_of_items` from scope and writes to `multiply` in scope [2:L143-160].

**Key invariants enforced by `basic_node_validator`** [2:L63-105]:
- `name` and `out` must be distinct (no FuncNode can write to a variable named after itself)
- All names must be valid Python identifiers
- All `bind` keys must correspond to actual parameters of `func`
- No duplicate names across `name`, `out`, and `bind.values()`

### 2.2 The Bipartite Graph Structure

The graph has two node types:
- **FuncNodes**: Computation vertices (drawn as boxes in graphviz)
- **VarNodes**: Data/variable vertices (drawn without shape -- just labels)

Edges always connect across types:
- VarNode -> FuncNode (input edge: the function consumes this variable)
- FuncNode -> VarNode (output edge: the function produces this variable)

This bipartite structure is constructed by `_func_nodes_to_graph_dict` [2:L553-560], which builds an adjacency dict `{node: [adjacent_nodes]}` from the FuncNode specifications. The graph is represented as a plain Python `dict` -- no special graph object.

### 2.3 DAG: The Callable Graph

`DAG` [3] is a `@dataclass` that takes an iterable of `FuncNode | Callable` and produces a callable whose:
- **Signature** is inferred from the root nodes (inputs not produced by any FuncNode)
- **Return value** is extracted from the leaf nodes (outputs not consumed by any FuncNode)
- **Execution** follows topological order through the graph

**Construction** (`__post_init__` [3:L502-527]):
1. Convert raw callables to FuncNodes via `ensure_func_nodes`
2. Build adjacency graph via `_func_nodes_to_graph_dict`
3. Topologically sort to get execution order
4. Separate into `func_nodes` and `var_nodes`
5. Compute `__signature__` from root nodes using `src_name_params`
6. Identify `roots` (input var nodes) and `leafs` (output var nodes)

**Execution** (`_call` [3:L573-584]):
1. Map positional and keyword arguments to a `scope` dict
2. Iterate through `func_nodes` in topological order, calling `call_on_scope`
3. Extract leaf values from scope using `extract_output_from_scope`

### 2.4 Mesh and Slabs

`Mesh` [2:L498-505] is a minimal dataclass holding `func_nodes` with a `synopsis_string` method. It is currently more of a placeholder than a fully realized abstraction.

`Slabs` [4] is a streaming variant: an iterator that repeatedly creates fresh scopes, calls components (analogous to FuncNodes), and yields the populated scope as a "slab." It supports:
- Context management (entering/exiting component contexts)
- Exception handling for graceful iteration termination
- Conversion to/from DAG

### 2.5 The Graph Toolkit (itools)

`itools.py` [5] provides pure-function graph algorithms operating on `Mapping[N, Iterable[N]]` representations:
- `topological_sort`: DFS-based topological ordering
- `root_nodes`, `leaf_nodes`: Identify boundary nodes
- `ancestors`, `descendants`: Reachability queries
- `has_cycle`: Cycle detection via DFS with recursion stack
- `edge_reversed_graph`: Edge inversion
- Various traversal utilities (`successors`, `predecessors`, `children`, `parents`)

This module is deliberately agnostic to `FuncNode` or `DAG` -- it works on any adjacency mapping, making it reusable beyond `meshed` itself.


## 3. Implicit Wiring by Name Convention

### How It Works

When a user passes plain functions to `DAG`, meshed infers the graph topology from argument names:

```python
def this(a, b=1):
    return a + b
def that(x, b=1):
    return x * b
def combine(this, that):
    return (this, that)

dag = DAG((this, that, combine))
```

The naming convention:
1. Each function `f` gets a FuncNode with `name = f.__name__ + "_"` and `out = f.__name__` (via `underscore_func_node_names_maker` [2:L23-61])
2. The `bind` is initialized to `{param: param}` for each parameter (identity mapping -- parameter names ARE variable node names)
3. Since `combine` has parameters named `this` and `that`, these match the `out` of the previous FuncNodes, creating edges

This is **Convention-over-Configuration** -- the naming convention *is* the wiring specification.

### Tradeoffs

**Advantages**:
- Zero-boilerplate DAG construction from functions that follow the convention
- Natural: if you name your function's parameter after the thing it consumes, the wiring "just works"
- Gradual escape hatch: `FuncNode(func, bind={...}, out='...')` for when conventions don't fit

**Disadvantages**:
- **Name collision**: If two functions share a parameter name (e.g., `b` in `this` and `that` above), they silently share the same input. This is sometimes intentional, sometimes not. The `parameter_merge` mechanism [6:L1105-1176] attempts to detect conflicts (mismatched defaults, kinds, annotations) but the default is strict, leading to errors that users find confusing.
- **Refactoring fragility**: Renaming a parameter changes the graph topology. This is invisible -- there's no static analysis to catch broken wires.
- **Non-standard naming pressure**: Functions must be named (or renamed) to match expected variable names, which conflicts with general-purpose naming conventions.
- **No explicit type checking**: The name match is purely string-based. There's no validation that the output type of one function is compatible with the input type of its consumer.

The `bind` mechanism provides the explicit escape hatch: `FuncNode(g, bind={'d': 'b'})` says "bind parameter `d` to variable node `b`." This is the explicit wiring that overrides convention.


## 4. Execution Model

### Current: Imperative Topological Traversal

The execution model is straightforward imperative evaluation [3:L573-584]:

1. Initialize `scope` from call arguments
2. For each FuncNode in topological order:
   - Extract inputs from scope using `bind`
   - Call `func`
   - Write result to `scope[self.out]`
3. Return leaf values

This is a single-threaded, eager, sequential evaluation. The `scope` is a plain `dict` that accumulates all intermediate results.

### Coupling Between Specification and Execution

The DAG class **partially decouples** specification from execution:

**Decoupled aspects**:
- The `scope` is an argument to `call_on_scope`, allowing custom MutableMappings (e.g., logging scopes, no-overwrite scopes [7:L32-68], or even remote storage via `dol`)
- The `extract_output_from_scope` is configurable (defaults to `extract_values` but can be overridden)
- `new_scope` factory is configurable
- `call_on_scope_iteratively` [3:L626-634] yields control after each FuncNode, enabling step-by-step debugging

**Coupled aspects**:
- Execution order is hardcoded to topological sort at construction time
- No mechanism for parallel execution of independent FuncNodes
- No lazy/pull-based evaluation (all nodes execute regardless of which outputs are needed)
- No reactive/push-based execution (changing an input doesn't propagate through)
- The scope dict is always in-memory

### Experimental Execution Strategies

The `scrap/` directory contains prototypes for alternative strategies:

- **`reactive_scope.py`** [8]: A `ReactiveScope` (MutableMapping) that, on `__setitem__`, triggers any FuncNodes whose dependencies are now satisfied. This is push-based reactive evaluation. The `ReactiveFuncNode` subclass adds dependency-checking to `call_on_scope`.

- **`cached_dag.py`** [7]: A `CachedDag` that provides pull-based, demand-driven evaluation with caching. Given a target variable name, it recursively computes only the necessary ancestors, caching intermediate results.

These are scrap/prototype code, not integrated into the main abstractions. They demonstrate the *possibility* of alternative execution strategies but are not yet generalized into a pluggable executor pattern.


## 5. Subgraph Projection and Partial Application

### __getitem__ (Graph Slicing)

`DAG.__getitem__` [3:L641-753] implements subgraph extraction via slice syntax:

```python
dag['g_':'h_']  # SubDAG from FuncNode g_ to FuncNode h_ (inclusive)
dag[['u', 'f']:['h', 'g']]  # SubDAG between multiple boundary nodes
```

The algorithm (`_subgraph_nodes` [3:L771-780]):
1. Compute all `descendants` of input nodes
2. Compute all `ancestors` of output nodes
3. Take the intersection (nodes reachable from inputs AND reaching outputs)
4. Filter to FuncNodes only
5. Order by original topological order

This is a form of **graph rewriting**: the original DAG is projected onto a subgraph defined by input/output boundaries, producing a new callable DAG with a new (derived) signature.

### partial (Graph-Level Currying)

`DAG.partial` [3:L784-847] performs partial application at the graph level:

```python
new_dag = dag.partial(c=3)  # Fix variable 'c' to value 3
```

The mechanism:
1. Map the provided arguments to variable node names via the DAG's signature
2. Use `partialized_funcnodes` [3:L399-416] to `functools.partial` each FuncNode whose `func` has parameters matching the bound variable names
3. Construct a new DAG from the partialized FuncNodes
4. Optionally remove bound arguments from the signature

This is analogous to currying but on a computation graph: fixing a VarNode to a value removes it from the DAG's input surface while preserving the graph topology.

### parametrized_dag_factory

`parametrized_dag_factory` [3:L1613-1678] splits a DAG into a "parameter computation" sub-DAG and a "main computation" sub-DAG, mimicking class initialization:

```python
factory = parametrized_dag_factory(dag, 'model_params')
configured_dag = factory(learning_rate=0.01, epochs=100)
result = configured_dag(data=my_data)
```

This realizes the **Strategy pattern at the graph level**: the same computation topology can be reconfigured by fixing different "parameter" VarNodes.


## 6. Relationship to Sibling Packages

### The Implicit Layered Architecture

`meshed` sits at the center of an ecosystem of packages that form an implicit layered architecture:

```
                          ┌──────────────────────────────────────────────────┐
    Application Layer     │  app_meshed  │  theremin  │  og  │  ef  │  uf   │
                          └────────────────────────┬─────────────────────────┘
                                                   │ uses
                          ┌────────────────────────▼─────────────────────────┐
    UI/Dispatch Layer     │  streamlitfront  │  dagapp  │  front  │  qh     │
                          └────────────────────────┬─────────────────────────┘
                                                   │ renders
                          ┌────────────────────────▼─────────────────────────┐
    Composition Layer     │               meshed (DAG, FuncNode)             │
                          └────────┬─────────────────────────────┬───────────┘
                                   │ extends                     │ extends
                          ┌────────▼─────┐               ┌───────▼──────────┐
    Linear Composition    │    lined     │               │  creek / slang   │
                          │  (pipelines) │               │   (streaming)    │
                          └──────────────┘               └──────────────────┘
```

**Notable downstream consumers**:
- **app_meshed**: A dedicated application package that wraps `meshed` DAGs into deployable services, demonstrating the "render target" pattern at the application level.
- **uf**: Uses `meshed.DAG` to compose functional workflows, particularly for audio/signal processing pipelines.
- **og**: Leverages meshed's graph manipulation for ontology/knowledge graph operations.
- **ef**: Employs meshed for composing experiment/evaluation flows.
- **theremin**: An audio synthesis application using meshed DAGs to wire signal-processing components (oscillators, filters, effects) into real-time audio pipelines.
- **know**: Data access/codec package that could serve as VarNode adapters in a polymorphic sourcing scheme.
- **posted**: Uses meshed for composing document-processing pipelines.

### lined (Linear Pipelines)

`lined` provides `Line` (a pipeline of functions) and `Pipeline`. `meshed` generalizes `lined` from linear chains to DAGs. The `line_with_dag` function in `composition.py` [9:L111-131] explicitly emulates a `Line` using `DAG`, and the DAG docstring notes "a pipeline is a subset of DAG" [3:L46].

### creek / slang (Streaming)

`creek` provides streaming primitives, `slang` provides audio-specific stream processing. `meshed.Slabs` [4] is the bridge -- it turns a DAG-like set of components into a streaming iterator. The `scrap/flow_control_script.py` [10] shows `meshed.DAG` being used with `creek.automatas` for stateful stream processing with control flow.

### front / streamlitfront / dagapp (UI Rendering)

These packages render DAGs as user interfaces. `dagapp` takes a `DAG` and generates a Streamlit or other UI where each root variable becomes an input widget and leaf variables become output displays [11]. `streamlitfront` and `front` provide more general UI rendering from function signatures.

This is the **"render targets" pattern**: the same `DAG` specification is "rendered" into different runtime forms -- a callable pipeline, a Streamlit app, or (via `qh`) an HTTP API.

### know / recode (Data Access)

`know` and `recode` provide data codecs and format conversions. These could serve as adapters in a polymorphic sourcing scheme where a VarNode could be populated from multiple data formats.

### dol / py2store (Storage)

`dol` provides the `MutableMapping` abstraction for storage. Since `meshed`'s scope is a `MutableMapping`, any `dol` store can serve as the scope, enabling distributed state. The `Slabs` docstring explicitly mentions `dol` and `py2store` for this purpose [4:L365-368].


## 7. The Mesh Vision

### What Is Implemented

As of the current codebase, `meshed` implements:

- **FuncNode and DAG**: Core bipartite computation graph with implicit wiring, topological execution, signature inference, and graphviz visualization.
- **Graph manipulation**: Subgraph projection (`__getitem__`), partial application (`partial`), function substitution (`ch_funcs`), node renaming (`ch_names`), edge addition (`add_edge`), DAG union (`__add__`).
- **Code round-tripping**: `code_to_dag` (parse Python functions into DAGs via AST) and `dag_to_code` (generate Python code from DAGs) via `makers.py` [12].
- **Slabs**: Streaming execution of DAG-like components with exception handling.
- **Serialization stubs**: `fnode_to_jdict` / `jdict_to_fnode` and `dag_to_jdict` / `jdict_to_dag` in `makers.py` [12:L783-829].
- **Caching / LazyProps**: `LazyProps` metaclass-like pattern for cached properties derived from DAG relationships [13].
- **Graph algorithms**: A full toolkit in `itools.py` for adjacency-mapping-based graph operations.

### Open Design Tensions

The GitHub issues and discussions [16] reveal several actively debated design tensions:

**Identity model** (issues #20, #29, #40, #41): The most pervasive open question. `FuncNode.name` serves as both human label and internal ID, but these are different concerns. Issue #20 proposes renaming to `func_id`; issue #29 proposes unifying `name` and `out` into a single `_id` (since a FuncNode has exactly one output, the node identity and its output identity could merge). Issue #40 reports that name unicity is not enforced -- two different functions with the same name silently create a broken graph.

**`__getitem__` semantics** (issues #22, #42; discussion #60): The sub-DAG slicing API has confusing inclusive/exclusive behavior for func nodes vs. var nodes. Discussion #60 asks "What do we actually want out of `DAG.__getitem__`?" -- the author acknowledges the current semantics are surprising even to themselves. Issue #42 questions whether `dag[k]` (single key, not a slice) should work.

**Signature comparison strictness** (issues #44, #45; discussion #52): `ch_funcs` was initially too permissive, then too strict on signature matching. The resolution direction: accept a user-specifiable comparison function, but this remains unresolved. Related to `i2` issues #43 and #50 on signature binary comparison design.

**Flow control in a pure-function DAG** (discussion #59): Conditionals and loops don't fit neatly into the acyclic, pure-function model. Discussion #59 proposes `CondNode` / `DAGX` extending DAG with conditional nodes. Discussion #70 (Mesh Components) argues instead for keeping control flow inside wrapped functions, not in the graph structure itself.

**Scope as extension point** (discussions #49, #64): The `scope` MutableMapping is recognized as a powerful extension point (custom backends, message buses, caching). Discussion #64 explicitly states the declarative graph enables flexible execution models (data parallelism, task parallelism, distributed) -- but the current implementation only supports centralized imperative execution.

### What Is Aspirational

Based on scrap code, discussions, and TODOs, the broader "mesh" vision includes:

1. **Reactive Execution** (`scrap/reactive_scope.py` [8]): A scope that triggers downstream computation on write. The prototype exists but cache invalidation (noted as TODO) and cycle handling are unsolved.

2. **Pull-Based / Lazy Execution** (`scrap/cached_dag.py` [7]): Demand-driven evaluation with caching. The prototype works but has known issues with cache consistency when inputs change.

3. **Heterogeneous Node Types**: Beyond pure functions, nodes could be:
   - **Storage interfaces** (Mapping/MutableMapping) -- reading from or writing to persistent stores
   - **Control flow nodes** (switch-case, conditionals) -- the `flow_control_script.py` [10] prototype uses `SimpleSwitchCase` with `creek.automatas`
   - **UI event sources** -- generating values from user interaction
   - **Stream processors** -- nodes that consume/produce iterators

4. **Polymorphic Sourcing**: A VarNode having multiple potential providers, with adapter chains resolving format mismatches. The `@provides` decorator [6:L148-189] hints at this but no resolution mechanism exists.

5. **Multiple Render Targets**: The same mesh rendered as a callable, REST API, CLI, Streamlit app, MCP tool surface. Partially realized through the `dagapp` and `front` packages but not formalized.

6. **Annotations-to-Meshes** (`scrap/annotations_to_meshes.py` [14]): Deriving mesh structure from type annotations on methods, generating Protocol classes from `Callable` type specifications. This addresses the type-safety gap.

7. **Dask Integration** (`scrap/dask_graph_language.py`): Translation between meshed's graph representation and Dask's graph format.

8. **Computational Path Resolution** (discussion #71): A `mk_func(objs, inputs, outputs)` that resolves computational paths across collections of DAGs sharing variable identifiers. This generalizes meshed from single-DAG execution to multi-DAG routing and dispatch.

9. **Visual Block Programming** (discussion #50): A two-way live editor for creating, connecting, and annotating nodes, with import/export to JSON, DOT, and Mermaid formats. This is the "no-code GUI" aspiration described in discussion #68.

10. **Mesh Components Library** (discussion #70): Registerable, composable DAG-templates with special operating interfaces (creation, viewing, attaching to existing DAGs). Proposed components include `if_then_else`, `switch_case`, `extractor`, `tee_cache`, `crudifier`, and `gurglers`.


## 8. Design Patterns Inventory

### 8.1 Convention-over-Configuration

**Where**: The implicit wiring system in `ensure_func_nodes` [2:L534-547] and `underscore_func_node_names_maker` [2:L23-61].

**How**: Function parameter names serve as implicit edge definitions. No explicit configuration required when naming conventions are followed.

### 8.2 Strategy Pattern (Execution Level)

**Where**: The `call_on_scope` method accepts a `MutableMapping` as scope; `new_scope` factory and `extract_output_from_scope` are configurable on `DAG`; `parameter_merge` controls conflict resolution.

**How**: Different execution semantics can be injected by providing different scope implementations (logging scope, no-overwrite scope, reactive scope).

### 8.3 Adapter Pattern

**Where**: `FuncNode.bind` [2:L132-175] -- remaps function parameter names to variable node names.

**How**: A function with parameter `x` can consume a variable named `item_price` via `bind={'x': 'item_price'}`. The FuncNode adapts the function's interface to the graph's namespace.

### 8.4 Composite Pattern

**Where**: `DAG.__add__` [3:L1243-1255] -- union of DAGs.

**How**: Two DAGs can be combined into a larger DAG. `DAG.__iter__` yields FuncNodes, enabling `DAG([*dag1, *dag2, extra_func])`.

### 8.5 Facade Pattern

**Where**: `DAG` itself is a facade over the graph construction, topological sorting, signature inference, and scope-based execution.

**How**: The user provides functions and gets back a callable with an inferred signature, hiding all the graph machinery.

### 8.6 Graph Rewriting

**Where**: `DAG.__getitem__` [3:L641-753] (subgraph projection), `DAG.partial` [3:L784-847] (graph-level currying), `ch_funcs` [3:L1790-1865] (function substitution), `ch_names` [3:L1870-1964] (variable renaming), `add_edge` / `add_edges` [3:L1260-1396].

**How**: Each operation produces a new DAG by transforming the graph structure or its components. The graph is treated as an immutable specification that can be derived into new specifications.

### 8.7 Dependency Injection via Name-Matching

**Where**: The entire implicit wiring system. Also `code_to_dag` [12] which parses AST to extract dataflow and uses `func_src` mapping to inject implementations.

**How**: Functions are "injected" into FuncNode slots based on name resolution. `ch_funcs` enables post-hoc function replacement. `code_to_dag` with `func_src` mapping separates the topology specification (the code structure) from the function implementations.

### 8.8 Topology-Driven Dispatch

**Where**: The sibling packages `dagapp`, `front`, `streamlitfront` which render DAGs as UIs.

**How**: The graph topology determines the UI layout -- root nodes become input widgets, leaf nodes become output displays, intermediate nodes may become visible intermediate results. The dispatch target (Streamlit, Flask, CLI) is selected independently of the topology.

### 8.9 Template Method (via `code_to_dag`)

**Where**: `code_to_dag` [12:L559-580] with `use_place_holder_fallback=True`.

**How**: The AST-parsed code defines the template (topology + calling convention), and concrete implementations are injected later via `func_src`. Placeholder functions generate descriptive strings, enabling DAG construction and testing before real implementations exist.

---

## REFERENCES

[1] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/__init__.py` -- Package entry point and public API.

[2] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/base.py` -- FuncNode, Mesh, validation, visualization primitives.

[3] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/dag.py` -- DAG class, graph manipulation, `ch_funcs`, `ch_names`.

[4] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/slabs.py` -- Slabs streaming execution.

[5] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/itools.py` -- Graph algorithms on adjacency mappings.

[6] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/util.py` -- Utilities: naming, validation, parameter merging, extraction.

[7] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/cached_dag.py` -- Pull-based cached evaluation prototype.

[8] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/reactive_scope.py` -- Reactive push-based evaluation prototype.

[9] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/composition.py` -- `line_with_dag`, `suffix_ids`.

[10] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/flow_control_script.py` -- Stateful control flow with DAG + creek automata.

[11] Cross-repo: `dagapp`, `streamlitfront`, `front` packages render DAGs as UIs.

[12] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/makers.py` -- `code_to_dag`, `code_to_fnodes`, serialization stubs, `triples_to_fnodes`.

[13] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/caching.py` -- `LazyProps`, cached property machinery.

[14] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/annotations_to_meshes.py` -- Type-annotation-driven mesh construction prototype.

[15] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/README.md` -- Package README with usage examples.

[16] https://github.com/i2mint/meshed -- GitHub repository, issues and discussions.
