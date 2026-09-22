# meshed

Link Python functions up into callable aggregate objects (DAGs, pipelines) from a
collection of objects and a composition policy. `FuncNode` is the unit meshed
assembles into `DAG`s. Legacy-packaged (`setup.cfg`/`setup.py`, no
`pyproject.toml`) but depended upon across the fleet — see Dependents below.

## Module map (`meshed/`)

- `base.py` — `FuncNode`, `compare_signatures`: the unit of computation meshed
  assembles into DAGs (name = identity in the network + signature).
- `dag.py` — `DAG`, `ch_funcs`, `ch_names`: the core DAG-building machinery.
- `itools.py` — graph operations over adjacency `Mapping`s: `random_graph`,
  `topological_sort`, etc. — usable standalone from the function-DAG layer.
- `makers.py` — `code_to_dag`, `code_to_fnodes`: build meshed objects from code.
- `composition.py` — specific compositions of `FuncNode`/`DAG`.
- `components.py` — ready-made extraction components for meshed graphs (giving a
  DAG node the `__name__`/signature it needs to identify its var node).
- `slabs.py` — `Slabs`: dicts holding stream data for a time interval.
- `caching.py` — attach `functools.cached_property` to a class from plain functions.
- `tools.py` — `mk_hybrid_dag` and friends — **optional integrations** (needs
  `http2py`/`extrude`, not installed by the bare package; see Tests below).
- `util.py` — function-wrapping/naming/small-data helpers the DAG machinery
  leans on: `iterize`, `ConditionalIterize`, `instance_checker`,
  `replace_item_in_iterable`, `parameter_merger`, `provides`, `Pipe`.
- `viz.py` — visualization utilities (needs `graphviz`/`networkx`/`ipywidgets`).

## Tests & lint (verified)

```bash
uv venv .venv && uv pip install -e . pytest
.venv/bin/pytest meshed/ --ignore=meshed/examples --ignore=meshed/scrap \
  --doctest-modules -q
```
With only the base install: 170 passed, 1 skipped, **1 failed**
(`test_meshed_tools.py::test_hybrid_dag` — `ModuleNotFoundError: http2py`/`extrude`).
This is expected, not a regression: `setup.cfg`'s `tests_require` lists
`http2py, extrude, graphviz, networkx, ipywidgets, ipython` separately from
`install_requires` (`i2` only) — install `tests_require` for a fully green run.

CI (`.github/workflows/ci.yml`) is the legacy `i2mint/isee` pipeline: installs
from `setup.cfg`, then **`pylint-validation`** (only `missing-module-docstring`
enabled, `tests`/`examples`/`scrap` ignored — every module needs a top-level
docstring, matching the user-level convention), then `pytest-validation`. Not
`ruff`/`wads` — `ruff check meshed/` reports hundreds of pre-existing findings
that are not what gates merges here.

## Docs

- `misc/docs/meshed_design_doc.md`, `meshed_improvement_ideas.md`, and the
  "Formal Foundations and Design Patterns for Declarative Object Composition" doc.
- `misc/Usage Examples.md`; notebooks under `misc/`.

## Dependents

`allude`, `dagapp`, `dotsci`, `extrude`, `front`, `guided`, `know`, `lookbook`,
`raglab-app`, `smart-cv`, `theremin`, `titbit`, `uf` import this package — check
their tests before changing `FuncNode`, `DAG`, or any public signature.
