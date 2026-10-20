# Naming helpers in `meshed.util`

Several `meshed` pieces need names that do not clash: the nodes of a `DAG`, the arguments of functions composed into one, copies of a DAG. A handful of small helpers in `meshed/util.py` do this.

| helper | what it does | used by |
|---|---|---|
| `find_first_free_name(prefix, exclude_names=(), start_at=2)` | `prefix` if free, else the first free `f"{prefix}__{i}"` (`ab`, `ab__2`, `ab__3`, ...) | `mk_func_name`, `arg_names` |
| `mk_func_name(func, exclude_names=(), *, strict=False, default="func")` | a free name for a function: unwraps `partial`s, gives lambdas a unique generated name, falls back to `default` (or raises `NameValidationError` when `strict`) | `FuncNode`, `DAG` |
| `arg_names(func, func_name, exclude_names=())` | `func`'s parameter names, with a clashing one renamed to the first free `f"{func_name}__{name}"` | `dag.py` (its own `arg_names`) |
| `numbered_suffix_renamer(name, sep="_")` | `item` -> `item_1` -> `item_2`: increments a trailing number, with no notion of what is taken | `DAG.copy`, `ch_names`, `composition` |
| `replace_item_in_iterable(iterable, condition, replacement, *, egress=None)` | replaces the items satisfying `condition` (a value or a predicate) while keeping the iterable's type | `DAG` rewiring |

```python
>>> from meshed.util import find_first_free_name, mk_func_name, numbered_suffix_renamer
>>> find_first_free_name('ab', ('ab', 'ab__2'))
'ab__3'
>>> mk_func_name(len, exclude_names=('len',))
'len__2'
>>> numbered_suffix_renamer('item_1')
'item_2'
```

## Relation to `i2` (decision, #91)

`i2` has `name_of_obj` (already used by `mk_func_name` to name objects that lack `__name__`) and `i2.multi_object.uniquely_named_objects`, which names a *collection* positionally (`list`, `_2`, `_3`, ...). They overlap in purpose but not in behaviour:

- `uniquely_named_objects` makes a clash-free name list for a whole collection and uses `_N` positional suffixes; `find_first_free_name` makes one name against a changing set of taken names and uses the `__N` suffix that appears in DAG node and argument names.
- Switching `meshed` to the `i2` function would change generated node and argument names, which dependents' tests and saved DAGs rely on.

So all five helpers stay in `meshed` (they encode `meshed`'s naming convention), and `name_of_obj` stays the shared piece from `i2`. Revisit only if `i2` grows a single-name, exclusion-set function with a configurable suffix format.
