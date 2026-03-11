# meshed: Improvement Critique and Architectural Roadmap

**A Candid Assessment of What Works, What Doesn't, and What Could Be**

---

## 1. Code Quality and Hygiene

### 1.1 The DAG Class: Approaching God-Object Territory

`dag.py` is 1980 lines [1]. The `DAG` class itself handles:
- Graph construction and topological sorting
- Signature inference and parameter merging
- Scope-based imperative execution
- Sub-DAG extraction (`__getitem__`)
- Partial application (`partial`)
- Function substitution (`ch_funcs`)
- Node renaming (`ch_names`)
- Edge manipulation (`add_edge`, `add_edges`)
- Visualization (`dot_digraph`, `dot_digraph_body`, `dot_digraph_ascii`)
- Debugging (`debugger`)
- Iterator protocol (`__iter__`)
- Union (`__add__`, `__radd__`)
- Code generation (`to_code`)
- Serialization support (via `synopsis_string`)

This is too many responsibilities for a single class. The graph construction/topology concern, the execution concern, the manipulation/rewriting concern, and the visualization concern should be separated. At minimum:
- A `DagGraph` or similar for topology, roots, leafs, subgraph operations
- `DAG` as the callable facade that composes a `DagGraph` with an execution strategy
- Visualization as standalone functions (partially done in `viz.py` but `DAG` still owns `dot_digraph`)

### 1.2 Duplicated Utility Functions

Several functions are duplicated between `dag.py` and `util.py`:
- `find_first_free_name` exists in both `dag.py:211` and `util.py:680`
- `mk_func_name` exists in both `dag.py:234` and `util.py:692`
- `arg_names` exists in both `dag.py:270` and `util.py:702`
- `named_partial` exists in both `dag.py:288` and `util.py:720`

These are not aliases -- they are independent implementations that could diverge. This needs consolidation.

### 1.3 The `is_func_node` Problem

`is_func_node` [2:L563-583] uses a deliberately weak check (matching `__name__` in the MRO) instead of `isinstance` because FuncNode's class identity can change during development (module reloading). The code itself calls this a TODO:

```python
# TODO: Replace with isinstance(obj, FuncNode) is this when development stabalizes
#  See: https://github.com/i2mint/meshed/discussions/57
```

Discussion #57 [11] documents this explicitly: "Make `isinstance(obj, FuncNode)` work!!" The workaround exists because module reloading creates different `FuncNode` classes with the same name. This is a development-time workaround that leaked into production code. It introduces fragility (any class named "FuncNode" will pass) and should be addressed.

### 1.4 Commented-Out Code and Stale TODOs

- `base.py:235-255`: An entire `__init__` method commented out with a note about Python 3.10
- `dag.py:1112-1138`: Commented-out alternative implementation of `ch_funcs`
- `dag.py:1827`: `# TODO: Optimize (for example, use self._func_node_for)`
- `dag.py:1785-1788`: Multiple TODOs indicating the author considers `ch_funcs` to have a "terrible" interface and code
- Over 60 TODOs across the codebase, many stale

### 1.5 Inconsistent Error Handling

The codebase mixes `assert` statements (which can be disabled with `-O`) with proper exception raising:
- `base.py:101-105`: Uses `assert` for bind key validation
- `dag.py:206`: Uses `assert` for set containment
- `itools.py:301`: Uses `assert` for type checking

`assert` should be reserved for invariants during development. User-facing validation should always use explicit exceptions.

### 1.6 Test Coverage

Tests are decent for the core DAG functionality but missing for:
- Edge cases in `Slabs` streaming
- `CachedDag` (noted in `cached_dag.py:345` with a broken test: `assert c("f" == 2)` which is always truthy due to the string `"f"` being compared to `2`)
- `ReactiveScope` (only doctest examples, no unit tests)
- Serialization (`fnode_to_jdict`, `dag_to_jdict`)
- The `makers.py` code-round-tripping (limited tests for complex cases like tuple unpacking)

### 1.7 The `__call__` Deprecation in FuncNode

`FuncNode.__call__` [2:L360-366] raises `DeprecationWarning` but does so via `raise`, not `warnings.warn`. This means calling a FuncNode directly crashes with an exception, not a warning. The commented-out `warn(...)` suggests this was intentional but the docstring says "Deprecated" which implies it should still work with a warning.


## 2. Abstraction Gaps

### 2.1 The Identity Model Problem

Issues #20, #29, #40, and #41 [11] reveal the most pervasive design debt: `FuncNode.name` conflates internal graph ID with human-readable label. Issue #20 proposes renaming to `func_id`; issue #29 proposes unifying `name` and `out` into a single `_id`. Issue #40 reports a regression: name unicity is not enforced, so two different functions with the same name create a silently broken graph. Issue #17 shows the user pain: reusing the same function requires renaming, which is confusing since the *function* hasn't changed.

This cluster of issues points to a fundamental modeling gap: the identity of a computation node, its human label, and its output variable are three different concerns currently squeezed into two fields (`name`, `out`).

### 2.2 No Explicit VarNode Type

Variable nodes are bare strings throughout the codebase. There is no `VarNode` class despite the name being used conceptually everywhere. This means:
- No place to attach metadata to variables (type, schema, description, default provider)
- No way to distinguish "this string is a variable name" from "this string is something else"
- No hook for type validation, format negotiation, or polymorphic sourcing

A `VarNode` protocol or dataclass would enable:
```python
@dataclass
class VarNode:
    name: str
    type_hint: type = Any
    description: str = ""
    providers: list[FuncNode] = field(default_factory=list)
    consumers: list[FuncNode] = field(default_factory=list)
```

### 2.3 Specification vs. Execution Not Cleanly Separated

The `DAG` class conflates the graph specification with the execution strategy:
- `_call` hardcodes scope-based imperative execution
- `call_on_scope` is baked into `DAG`, not injected
- The topological ordering is computed at construction and embedded in `self.func_nodes`

There should be:
- A `DagSpec` (pure specification: nodes, edges, types) that is immutable and serializable
- An `Executor` protocol that takes a `DagSpec` and a set of inputs and produces outputs
- `DAG` as a convenience that binds a `DagSpec` with a default `ImperativeExecutor`

### 2.4 No Edge First-Class Object

Edges are implicit in the `bind` dictionaries and the adjacency mapping. There is no `Edge` type that could carry metadata (e.g., type constraints, transformation functions, optional/required status). This makes it impossible to:
- Annotate edges with type converters
- Mark edges as optional (graceful degradation if source unavailable)
- Attach debugging/logging to specific data flows

### 2.5 `scope` Coupling

The entire execution model depends on a shared mutable `dict` as the communication mechanism between FuncNodes. This is simple and effective for single-threaded sequential execution, but it:
- Prevents parallel execution (race conditions on shared dict)
- Makes it impossible to track data provenance (who wrote what, when)
- Conflates inputs, intermediates, and outputs in the same namespace
- Has no mechanism for scope "views" (showing a FuncNode only the variables it's allowed to see)


## 3. The Multi-Source / Polymorphic Sourcing Problem

### The Problem

A variable like `waveform` might be sourced from:
- A raw NumPy array (already in the expected format)
- A WAV file path (needs file reading + decoding)
- A URL (needs HTTP fetch + decoding)
- A byte stream (needs decoding only)

Currently, meshed has no mechanism for this. A VarNode is either populated by a single FuncNode's output or provided as a root input. There is no "resolution" logic.

### Proposed Design

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class VarNodeSource(Protocol):
    """A potential provider for a VarNode."""
    def can_provide(self, available_vars: dict) -> bool:
        """Whether this source can provide the value given current state."""
        ...

    def provide(self, available_vars: dict) -> Any:
        """Compute the value."""
        ...

    @property
    def priority(self) -> int:
        """Higher priority sources are tried first."""
        ...

@dataclass
class AdaptiveVarNode:
    name: str
    sources: list[VarNodeSource] = field(default_factory=list)
    type_hint: type = Any

    def resolve(self, scope: dict) -> Any:
        """Try sources in priority order, return first successful."""
        for source in sorted(self.sources, key=lambda s: -s.priority):
            if source.can_provide(scope):
                return source.provide(scope)
        raise ResolutionError(f"No source could provide '{self.name}'")
```

This is the **Chain of Responsibility** pattern applied to data sourcing. Each source is an adapter that can optionally provide a value if its preconditions are met.

The `@provides` decorator [3:L148-189] is a step in this direction -- it declares what variable a function can provide. But there's no corresponding resolution mechanism that uses this declaration.


## 4. Heterogeneous Node Types

### Current State

All nodes are either FuncNodes (callables) or bare strings (VarNodes). The `Slabs` class [4] hints at heterogeneity -- its components can be zero-argument callables (data sources) alongside normal functions. But there's no formal node type hierarchy.

### Proposed Architecture

Define a `Node` protocol hierarchy:

```python
class Node(Protocol):
    """Base protocol for all nodes in a mesh."""
    name: str

    def dot_lines(self, **kwargs) -> Iterable[str]:
        """GraphViz representation."""
        ...

class ComputeNode(Node):
    """A node that computes an output from inputs."""
    bind: dict[str, str]
    out: str

    def call_on_scope(self, scope: MutableMapping) -> Any: ...

class StorageNode(Node):
    """A node backed by a Mapping (read) or MutableMapping (read/write)."""
    store: Mapping

    def read_to_scope(self, scope: MutableMapping, key: str) -> Any: ...
    def write_from_scope(self, scope: MutableMapping, key: str) -> None: ...

class SourceNode(Node):
    """A node that produces values without inputs (e.g., sensor, timer, UI event)."""
    def produce(self) -> Any: ...

class ControlNode(Node):
    """A node that routes data based on conditions."""
    condition: Callable
    branches: dict[Any, str]  # condition_result -> target_var_node
```

`FuncNode` would implement `ComputeNode`. The `DAG` (or `Mesh`) would accept any `Node`, dispatching to the appropriate execution logic based on the protocol it satisfies.

The key architectural choice is whether to use **inheritance** (subclassing FuncNode) or **protocols** (structural typing). Given meshed's Python-idiomatic philosophy, **protocols** are preferable:
- No coupling to a specific base class
- Existing objects that happen to satisfy the interface work without modification
- Aligns with `dol`'s approach of wrapping existing objects with Mapping interfaces


## 5. Execution Strategy Decoupling

### Proposed Executor Protocol

```python
class Executor(Protocol):
    """Executes a DAG specification."""

    def execute(
        self,
        spec: DagSpec,
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the DAG with given inputs, return outputs."""
        ...

class ImperativeExecutor:
    """Current behavior: sequential topological traversal."""
    def execute(self, spec, inputs):
        scope = dict(inputs)
        for node in spec.topological_order():
            node.call_on_scope(scope)
        return {k: scope[k] for k in spec.leaf_names}

class ParallelExecutor:
    """Execute independent nodes concurrently."""
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def execute(self, spec, inputs):
        # Group nodes by "level" (all at same level are independent)
        # Execute each level in parallel
        ...

class LazyExecutor:
    """Pull-based: only compute what's needed for requested outputs."""
    def execute(self, spec, inputs, *, requested_outputs=None):
        # Recursively resolve only ancestors of requested outputs
        ...

class ReactiveExecutor:
    """Push-based: propagate changes from modified inputs."""
    def execute(self, spec, inputs, *, changed_keys=None):
        # Only recompute descendants of changed keys
        ...
```

### Comparison with Existing Frameworks

| Framework | Scope | Executor Model |
|-----------|-------|----------------|
| **Dask** | Distributed compute | Lazy graph + scheduler (synchronous, threaded, multiprocessing, distributed) |
| **Prefect** | Workflow orchestration | Task runners (Sequential, Concurrent, Dask, Ray) |
| **Airflow** | Batch scheduling | Executor plugins (Local, Celery, Kubernetes) |
| **Ray** | Distributed compute | Actor/task model with DAG scheduling |
| **meshed** (proposed) | In-process composition | Pluggable executor on lightweight graph spec |

The key difference: meshed's graph is **fine-grained** (individual function calls, not coarse tasks) and **in-process** (not distributed). The executor abstraction should be simpler than Dask's scheduler but follow the same principle of separating what to compute from how to compute it.

**Precedent**: Issue #65 and merged PR #69 [11] demonstrate a "local+cloud scheduler flexibility" POC using `py2http` / `http2py` to offload individual DAG func nodes to cloud workers. This was accomplished by wrapping functions (via `ch_funcs`) rather than changing the executor -- validating the approach but also showing its limits (function wrapping is per-node, not per-execution-strategy).


## 6. Serialization and Persistence

### Current State

`makers.py` has `fnode_to_jdict` / `jdict_to_fnode` and `dag_to_jdict` / `jdict_to_dag` [5:L783-829]. These serialize the topology (name, bind, out, func_label) but NOT the function implementations -- `jdict_to_fnode` requires a `jdict_to_func` callable to reconstruct functions.

`code_to_dag` / `dag_to_code` [5] provides a code-level round-trip: parse Python code into a DAG, or generate Python code from a DAG.

### What's Missing

1. **No canonical serialization format**. The JSON stubs exist but aren't standardized or documented.

2. **No function serialization strategy**. Functions can't be serialized in general, but the topology CAN be serialized with function references (import paths or registry keys). This is what `code_to_dag`'s `func_src` parameter does, but it's not formalized.

3. **No versioning or diffing**. If meshes are to be the IR, you need to version them and compute diffs ("what changed between v1 and v2 of this pipeline?").

### Proposed Serialization Format

```yaml
# mesh_spec.yaml
name: audio_pipeline
version: "1.0"
nodes:
  - name: read_audio
    func_ref: "mypackage.io.read_wav"
    bind: {path: audio_path}
    out: waveform
  - name: extract_features
    func_ref: "mypackage.features.mfcc"
    bind: {signal: waveform, sr: sample_rate}
    out: features
  - name: classify
    func_ref: "mypackage.models.predict"
    bind: {x: features}
    out: prediction
inputs:
  audio_path: {type: str, description: "Path to WAV file"}
  sample_rate: {type: int, default: 16000}
outputs:
  prediction: {type: str}
```

The `func_ref` is a dotted import path. Deserialization resolves these via import or a registry. This is similar to Airflow's operator import model.

For the "mesh as IR" vision, consider adopting or aligning with an existing IR:
- **ONNX**: For ML-specific meshes
- **Apache Arrow / Substrait**: For data processing meshes
- **Custom YAML/JSON**: For general-purpose meshed IR (most likely)


## 7. Type System and Contracts

### The Problem

Currently, wiring validation is purely name-based. This code silently connects incompatible types:

```python
def produce_int(x) -> int:
    return int(x)

def consume_str(produce_int: str) -> str:  # expects str, gets int!
    return produce_int.upper()

dag = DAG([produce_int, consume_str])  # No error at construction
dag("42")  # Crashes at runtime: AttributeError: 'int' has no 'upper'
```

### Proposed: Type Contracts on VarNodes

```python
from typing import get_type_hints

def validate_dag_types(dag: DAG) -> list[TypeError]:
    """Check that producer output types match consumer input types."""
    errors = []
    for func_node in dag.func_nodes:
        hints = get_type_hints(func_node.func)
        # Check return type against consumers
        return_type = hints.get('return', Any)
        for consumer_fn in dag.consumers_of(func_node.out):
            consumer_hints = get_type_hints(consumer_fn.func)
            for param, var_name in consumer_fn.bind.items():
                if var_name == func_node.out:
                    expected_type = consumer_hints.get(param, Any)
                    if not issubclass(return_type, expected_type):
                        errors.append(TypeError(
                            f"{func_node.out}: {func_node.name} produces "
                            f"{return_type} but {consumer_fn.name} expects "
                            f"{expected_type} for param '{param}'"
                        ))
    return errors
```

This could be an optional validation step at DAG construction time (off by default for progressive disclosure, on for production).

For a more expressive system, consider **schema declarations** on VarNodes:

```python
@dataclass
class TypedVarNode:
    name: str
    schema: type | Schema  # Could be a type, Pydantic model, or JSON Schema

    def validate(self, value: Any) -> Any:
        """Validate and possibly coerce value to match schema."""
        ...
```


## 8. API Ergonomics

### 8.1 The "Hello World" Experience

The simplest possible meshed usage is excellent:

```python
from meshed import DAG

def add(a, b): return a + b
def mult(add, c): return add * c

dag = DAG([add, mult])
dag(1, 2, 3)  # (1+2)*3 == 9
```

This is clean and intuitive. The progressive disclosure principle is well-satisfied at this level.

### 8.2 The "Second Step" Problem

The moment you need to deviate from the naming convention, the complexity jumps sharply:

```python
from meshed import FuncNode, DAG

dag = DAG([
    FuncNode(func=my_adder, name='add_step', bind={'x': 'input_a', 'y': 'input_b'}, out='sum'),
    FuncNode(func=my_multiplier, name='mult_step', bind={'a': 'sum', 'b': 'factor'}, out='product'),
])
```

The `bind` dict is confusing: keys are the function's parameter names, values are the variable node names. This is backwards from what most people expect (they think `bind={'source_var': 'param_name'}`). The docstring explains this [2:L131-175] but it's a recurring source of confusion.

**Suggestion**: Offer a fluent builder API as an alternative:

```python
dag = (DAG.builder()
    .add(my_adder, name='add_step')
        .input('x', from_var='input_a')
        .input('y', from_var='input_b')
        .output('sum')
    .add(my_multiplier, name='mult_step')
        .input('a', from_var='sum')
        .input('b', from_var='factor')
        .output('product')
    .build())
```

### 8.3 The Underscore Naming Convention Is Confusing

By default, `FuncNode(add)` produces `name='add_'` and `out='add'`. The underscore suffix on the function node name while the output gets the clean name is surprising. It's done to avoid name collision between the function identity and its output variable, but users regularly find it confusing in `synopsis_string` output and when trying to reference nodes by name in `__getitem__` or `ch_funcs`. Issues #20 and #29 [11] are direct attempts to resolve this tension -- #29 proposes collapsing `name` and `out` into a single identifier, eliminating the underscore convention entirely.

### 8.4 `__getitem__` Semantics Are Surprising

Issues #22, #42, and discussion #60 [11] document widespread confusion about sub-DAG slicing. Discussion #60 quotes the author: the inclusive/exclusive semantics for func nodes vs. var nodes are confusing even to the designer. Issue #42 asks whether `dag[k]` (single key) should work -- currently it raises `AssertionError`. The fundamental question posed in #42: "Are we creating complexity because of a bad design -- the inconsistency -- or is the design choice a good one?"

### 8.5 Error Messages

Error messages are generally good but some are unnecessarily opaque:

- `parameter_merger` raises `ValidationError` with a long suggestion block [3:L1134-1148] that, while helpful, makes the actual error message hard to find.
- The cycle detection in `has_cycle` [6:L424-506] documents regret about the graph direction convention ("I regret this design choice") which, while honest, should be addressed rather than documented.
- `_func_node_args_validation` [2:L714-738] validates type but doesn't suggest corrections.

### 8.6 Missing `__repr__` for DAG

`DAG.__repr__` uses the default dataclass repr, which dumps all func_nodes inline:
```
DAG(func_nodes=[FuncNode(a,b -> f_ -> f), FuncNode(c,d -> g_ -> g), ...], name='arithmetic')
```

For large DAGs this is unreadable. A better repr would show the synopsis string or just node count:
```
DAG('arithmetic', 3 nodes: a,b,c,d -> f,g,h)
```

### 8.7 The `@code_to_dag` Decorator Pattern

`code_to_dag` doubling as a decorator (via `@double_up_as_factory`) is clever:

```python
@code_to_dag(func_src=locals())
def my_pipeline(x, y):
    a = step1(x)
    b = step2(a, y)
    c = step3(b)
```

But `func_src=locals()` is fragile and requires that all referenced functions are already in scope. The default behavior (placeholder functions) is confusing for newcomers who expect the DAG to actually compute. Documentation should make this distinction much clearer.


## 9. Concrete Recommendations

### Quick Wins (Days)

1. **Consolidate duplicated utilities**: Eliminate the `dag.py` copies of `find_first_free_name`, `mk_func_name`, `arg_names`, `named_partial`. Import from `util.py` instead.

2. **Replace `assert` with proper exceptions**: Audit all `assert` statements in non-test code and replace with `ValueError`, `TypeError`, or custom exceptions.

3. **Fix `is_func_node`**: Replace the MRO name-matching hack with `isinstance(obj, FuncNode)`. If development-time reloading is a concern, add a `_IS_FUNCNODE` sentinel attribute that FuncNode sets, and check for that. (See discussion #57 [11].)

3b. **Enforce name unicity at DAG construction**: Two FuncNodes with the same `name` or `out` should raise an error immediately. Currently they silently create a broken graph (issue #40 [11]).

4. **Improve `DAG.__repr__`**: Show name, node count, and root/leaf variables instead of dumping all FuncNodes.

5. **Fix broken test in `cached_dag.py:363`**: `assert c("f" == 2)` should be `assert c("f") == 2`.

6. **Clean up commented-out code**: Remove the commented `__init__` in `base.py:235-255` and the dead code in `dag.py:1112-1138`.

### Medium-Term (Weeks)

7. **Extract graph topology into separate class**: Create a `DagTopology` (or `DagSpec`) class that holds the graph, topological order, roots, and leafs. `DAG` composes this with execution logic.

8. **Introduce VarNode dataclass**: Even a simple `@dataclass class VarNode: name: str` would provide a place to attach type hints and metadata later.

9. **Add optional type checking at construction time**: Use `typing.get_type_hints` to validate that producer return types are compatible with consumer parameter types. Make it opt-in via a `validate_types=True` parameter.

10. **Standardize serialization**: Formalize the JSON serialization format in `makers.py`. Document it. Add round-trip tests.

11. **Add a `DAG.from_code` classmethod**: Make the `code_to_dag` pattern more discoverable as a DAG construction method rather than a separate module-level function.

12. **Fluent edge-building API**: Consider `DAG([f, g, h]).wire(f.output >> g.input('x'))` or similar syntax sugar for explicit wiring.

### Architectural Refactors (Months)

13. **Executor protocol**: Separate execution strategy from graph specification. Implement `ImperativeExecutor` (current behavior), `ParallelExecutor` (concurrent independent nodes), `LazyExecutor` (pull-based), and `ReactiveExecutor` (push-based) as pluggable strategies.

14. **Promote ReactiveScope**: Graduate `scrap/reactive_scope.py` into a proper execution strategy. Solve cache invalidation by tracking dependency versions.

15. **Heterogeneous node types via protocols**: Define `ComputeNode`, `StorageNode`, `SourceNode`, `ControlNode` protocols. Make `DAG` (or `Mesh`) accept any node satisfying the protocol.

16. **Polymorphic sourcing**: Implement `AdaptiveVarNode` with prioritized source resolution. Integrate with the `@provides` decorator.

17. **Mesh as IR**: Build a canonical serialization format (YAML/JSON) with function references. Add versioning, diffing, and a registry for function resolution.

18. **Formalize the render-target pattern**: Create a `Renderer` protocol that takes a `DagSpec` and produces a runtime form (callable, REST API, CLI, Streamlit app). Consolidate the scattered implementations in `dagapp`, `front`, `streamlitfront`, `qh`.

---

## REFERENCES

[1] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/dag.py` -- DAG class (1980 lines).

[2] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/base.py` -- FuncNode, validation, naming.

[3] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/util.py` -- Utilities, parameter_merger.

[4] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/slabs.py` -- Slabs streaming.

[5] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/makers.py` -- code_to_dag, serialization.

[6] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/itools.py` -- Graph algorithms.

[7] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/reactive_scope.py` -- Reactive prototype.

[8] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/cached_dag.py` -- Cached/lazy prototype.

[9] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/flow_control_script.py` -- Control flow.

[10] `/Users/thorwhalen/Dropbox/py/proj/i/meshed/meshed/scrap/annotations_to_meshes.py` -- Type-driven meshes.

[11] https://github.com/i2mint/meshed -- Issues and discussions.

[12] Cross-repo packages: `dagapp`, `streamlitfront`, `front`, `qh`, `lined`, `creek`, `slang`, `dol`.
