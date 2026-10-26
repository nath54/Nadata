# Cross-language benchmarks

One protocol, three implementations, saved results:

| Implementation | Runner | Measures |
|---|---|---|
| **C++** | `native/wrappers/tests/cpp/benchmark.cpp` (via `run.py --bench`) | the full engine through the C ABI (embedded runtime) |
| **Python** | `benchmarks/python_benchmark.py` | the full engine through `nadata.api` (the dev facade) |
| **SQLite** | `benchmarks/sqlite_benchmark.py` | raw row I/O — the "no engine" reference point |

```
python benchmarks/run_all.py            # writes results.md + results.json
python benchmarks/run_all.py --skip-cpp # python + sqlite only
```

Every implementation **verifies its own outputs** and emits
`check <name>: ok` lines; `run_all.py` exits non-zero if any check
fails, so a perf run can never mask a correctness break. Results land
in `benchmarks/results.md` (comparison tables) and
`benchmarks/results.json` (machine-readable, incl. env metadata).

## Protocol

Six workloads, same sizes everywhere (the db is modified in memory
only; a fresh fixture per run):

| Workload | ops | nadata | SQLite mirror |
|---|---|---|---|
| `load` | 20 | full project load | connection open + full-row read |
| `get_attribute` | 10000 | typed attribute read (tom.birthplace) | point SELECT by symbol |
| `set_attribute` | 10000 | typed attribute write (idempotent) | point UPDATE |
| `query_where_count` | 1000 | parse + evaluate `h.name == "Tom"` + count | indexed COUNT(*) |
| `create_instance` | 1000 | schema-checked insert into a new struct | plain INSERT |
| `call_function` | 10000 | `is_adult(tom)` through the engine | per-row UDF call |

Plus `mem:` peak-RSS lines (Windows + Linux; 0 on unsupported hosts).

## Why the Python facade can beat C++ on cheap ops (results.md)

The three columns do NOT do the same amount of work per call — run
`python benchmarks/decomposition.py` for the measured proof:

```
engine only (instance.get_attribute + unwrap):   0.20 us/op
python facade (+ symbol lookup, + wrapper):      0.83 us/op
C shim (facade call + UTF-8 + malloc + NdValue): 2.73 us/op
```

- The ENGINE work is identical for every language (0.2 us).
- The Python facade never leaves the Python heap: the result is
  already a `str`, so it pays only method dispatch (~0.6 us).
- The C/C++ path crosses the Python/C boundary EVERY call: PyObject_Call
  with a tuple, UTF-8 conversion, malloc, NdValue packaging (~2.5 us).

So "Python 2x faster than C++" means the boundary tax is ~1 us per
call — an excellent FFI price that buys hosts a Python-free process
and a stable ABI at ~500k reads/sec. It only matters in very hot
loops; for those, batch (pluck/aggregates) or stay in-engine.

## Fairness caveats (read before quoting numbers)

- **SQLite is a floor, not a competitor**: it does raw row I/O with no
  schema semantics, no expression evaluation, no interpreter. nadata's
  numbers include all of that — the gap is the price of the engine.
- `create_instance` on SQLite is a **durable** INSERT (in an open
  transaction); nadata's create is in-memory until `save_changes`.
- `call_function` maps to a per-row UDF — the closest SQLite
  equivalent of calling an engine function.
- The mirror sets `journal_mode=MEMORY` and `locking_mode=EXCLUSIVE`
  (single process); without the latter, Windows per-statement file
  locks dominate reads by ~50x. The `load` workload therefore runs
  last for SQLite (parser keys by name, order is irrelevant).
- C++/Python engine numbers should be close to each other; if they
  diverge, the C++ binding or the shim has overhead worth investigating.
