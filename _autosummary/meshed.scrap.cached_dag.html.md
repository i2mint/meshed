# meshed.scrap.cached_dag

Moved: `CachedDag` now lives in [`meshed.cached_dag`](meshed.cached_dag.html.md#module-meshed.cached_dag) (i2mint/meshed#36).

This module re-exports it for code that imported it from `meshed.scrap`.

### Functions

| `add`(a[, b])                                                      |                                                                                                                                                                                       |
|--------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`cached_dag_test`](#meshed.scrap.cached_dag.cached_dag_test)() | Covering issue [https://github.com/i2mint/meshed/issues/34](https://github.com/i2mint/meshed/issues/34) about "CachedDag.cache should be populated with inputs that it was called on" |
| `exp`(mult[, n])                                                   |                                                                                                                                                                                       |
| `mult`(x[, y])                                                     |                                                                                                                                                                                       |
| `subtract`(a[, b])                                                 |                                                                                                                                                                                       |

### meshed.scrap.cached_dag.cached_dag_test()

Covering issue [https://github.com/i2mint/meshed/issues/34](https://github.com/i2mint/meshed/issues/34)
about “CachedDag.cache should be populated with inputs that it was called on”
