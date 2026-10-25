# Wrapper tests — one contract, every language

Every wrapper (C++, Kotlin/JNA, Rust, JavaScript/Pyodide, ...) replays
the **same case ids** and the **same benchmark protocol**, so coverage
and performance are directly comparable across languages.

## Case ids

The canonical registry is `tests/test_abi_conformance.py` — every test
there carries a stable id (`@abi_case("nd_x/scenario")`) named after
the `nadata.h` function it exercises (the ABI 2 by-entity-id cases
live in `TestChoiceB`). A wrapper suite must:

1. Use the **same ids** (grep-able, diffable against the Python suite).
2. Use the **same fixtures/expectations** (see the table below).
3. Exit non-zero on failure and print `ok <case-id>` per passing case
   so any runner can parse results.

### Shared fixtures (from `tests/abi/fixtures.py`)

| Fixture | Meaning |
|---|---|
| `ex0_simple` (repo `./examples/ex0_simple`) | 3 Humans (tom, jean, louise), 1 white_cat; `count("Entity") = 4` (polymorphic), `total = 6`; symbols `tom`, `jean`; `is_adult` function; `Human.older(other)` method |
| `numeric_project` | 3 Boxes: weights 1.5/2.5/3.5, counts 2/3/5 — aggregation targets |
| `session_project` | `#init` seed + `#manual(v: Int = 7, name: Str = "x")` capture script |
| `dumped_ex0` | ex0 dumped to a fresh `.ndb` for `nd_open` roundtrips |

Status expectations map 1:1 to `tests/abi/contract.py` (e.g. unknown
symbol → NOTFOUND, bad `.ndb` → VALUE, `n ====` → SYNTAX).

## Layout

```
native/wrappers/
  cpp/nadata.hpp              header-only C++17 wrapper
  tests/
    README.md                 this contract
    cpp/
      test_conformance.cpp    case-id replay (same ids as Python)
      benchmark.cpp           speed + memory protocol below
      run.py                  builds & runs against the dev shim
```

Each future language adds `tests/<lang>/` with the same two binaries
(scripts for JVM/JS, cargo test for Rust).

## Benchmark protocol

`benchmark.cpp` is the reference implementation. Every language port
runs **the same workloads with the same sizes** and prints:

```
bench <name>: <ops> ops in <ms> ms => <ops/sec>
mem: start RSS <MB>, peak RSS <MB>, delta <MB>
```

| # | Workload | Ops | Notes |
|---|----------|-----|-------|
| 1 | `load` | 20 | full `Nadata.load(ex0)` + close |
| 2 | `get_attribute` | 10 000 | `tom.birthplace` |
| 3 | `set_attribute` | 10 000 | `tom.birthplace = "Lyon"` (idempotent write) |
| 4 | `query_where_count` | 1 000 | `query(Human).where_str("h", "h.name == \"Tom\"").count()` |
| 5 | `create_instance` | 1 000 | `Weapon(name = "Blade N")` — grows the db |
| 6 | `call_function` | 10 000 | `is_adult(tom)` |

Memory is **peak process RSS** (Windows `GetProcessMemoryInfo`,
Linux `VmHWM` from `/proc/self/status`, else `n/a`). Run the two
binaries back-to-back on the same machine for comparable numbers;
absolute values are machine-dependent, *ratios between languages* are
the signal.

## Running the C++ suite (host toolchain)

```bash
python native/wrappers/tests/cpp/run.py            # build + conformance
python native/wrappers/tests/cpp/run.py --bench    # build + benchmark
```

Requires the dev shim (`python native/build.py`) and the repo root as
cwd (fixtures are read from `./examples`).
