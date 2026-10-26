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
  java/nadata/                JNA binding (library + high-level API)
  kotlin/nadata/NadataExt.kt  idiomatic Kotlin layer over the JNA binding
  tests/
    README.md                 this contract
    cpp/
      test_conformance.cpp    case-id replay (same ids as Python)
      benchmark.cpp           speed + memory protocol below
      run.py                  builds & runs against the dev shim
    java/
      Conformance.java        case-id replay (same ids as Python/C++)
      Benchmark.java          speed + memory protocol (same workloads)
      run.py                  compiles + runs against the dev shim
    kotlin/
      Conformance.kt          case-id replay (same ids as Python/C++/Java)
      run.py                  downloads kotlinc, compiles + runs
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

C++ reference (Win11, same session as the Java numbers):

| Workload | ops/sec |
|---|---|
| load | 61 |
| get_attribute | 563,739 |
| set_attribute | 219,935 |
| query_where_count | 21,382 |
| create_instance | 108,906 |
| call_function | 18,252 |

## Running the Java suite (JDK 11+, JNA 5.14)

```bash
python native/wrappers/tests/java/run.py            # download JNA, compile, run
python native/wrappers/tests/java/run.py --bench   # benchmark (same protocol)
```

Reference numbers (same session as the C++ table, Win11, ops/sec):

| Workload | C++ | Java/JNA | ratio |
|---|---|---|---|
| load | 61 | 58 | 1.05x |
| get_attribute | 563,739 | 80,227 | 7.0x |
| set_attribute | 219,935 | 93,025 | 2.4x |
| query_where_count | 21,382 | 15,947 | 1.3x |
| create_instance | 108,906 | 34,254 | 3.2x |
| call_function | 18,252 | 15,214 | 1.2x |

The pattern is instructive: **coarse operations (load, query,
call_function) land within 5-30% of C++**, while the fine-grained
get/set pay the JNA proxy dispatch (~1-2µs/call) plus the per-call
32-byte block allocation. Peak RSS 131 MB (JVM baseline ~56 MB).
A JNA *direct-mapping* port could shave part of the fine-grained tax.

## Running the Kotlin suite (JDK 11+, kotlinc 2.0.21)

```bash
python native/wrappers/tests/kotlin/run.py            # download kotlinc, compile, run
python native/wrappers/tests/kotlin/run.py --skip-dl  # offline (cached)
```

The Kotlin wrapper (`native/wrappers/kotlin/nadata/NadataExt.kt`) is a
thin idiomatic layer **over the Java binding** — zero new native code:
operator `get`/`set` (`db["tom", "birthplace"]`), properties, `Long?`
instead of `OptionalLong` (`firstIdOrNull`, `getByNidOrNull`),
`getOrNull` (NOTFOUND -> null) and an AutoCloseable `using {}`.
Extensions compile to static methods calling the same Java methods,
so JVM-benchmark numbers are identical by construction (no separate
benchmark port needed). 76/76 cases green (the 72 Java ids plus
null-mapping helpers).

Note: `run.py` drives `kotlin-compiler.jar` directly through the host
JVM — the `kotlinc.bat` launcher mishandles quoted paths containing
spaces (e.g. `Cerisara Nathan`).

Requires `javac`/`java` on PATH (or `JAVA_HOME`), the dev shim
(`python native/build.py`) and the repo root as cwd. The JNA jar is
cached in `.native-cache/jna/`.

### Java coverage notes (72/72 ok)

All case ids from the C++ suite except:
- `nd_query_where_cb/*` — **intentionally absent**: JNA callbacks run
  on attached threads and cannot honour the shim's GIL contract;
  `whereStr` covers the same engine path.
- Some C++ cases were merged when they share one Java expression
  (e.g. dump/open roundtrip + file-must-not-exit guard).

Requires the dev shim (`python native/build.py`) and the repo root as
cwd (fixtures are read from `./examples`).
