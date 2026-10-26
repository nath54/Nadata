# The native ABI — host integration guide

The Nadata engine can be embedded into any host language through a
single stable C ABI (`native/nadata.h`), implemented by a shared
library that carries its own CPython runtime. This document is the
contract for host authors: calling conventions, ownership rules,
error handling, and the runtime layout.

For the Python facade being mapped, see `docs/api.md`.

---

## 1. What ships in a distribution

```
dist/native/<target>/
    nadata.dll | libnadata.so | libnadata.dylib   the shim (interpreter inside on Linux)
    nadata.lib                 Windows import library (for linking hosts)
    nadata.h                   the ABI — the only header hosts need
    nadata.hpp                 header-only C++17 wrapper (nadata::)
    nadata/                    the engine package (pure Python, zero deps)
    runtime/                   the pinned CPython (python-build-standalone)
    python312.dll              Windows copy next to the shim
```

The shim locates `runtime/` and `nadata/` **relative to the library's
own path** (`PyConfig.home` is set internally). Hosts never set
`PYTHONHOME`, `PYTHONPATH`, or registry keys. The one dev-only escape
hatch is `NADATA_HOME` — set it to a repository checkout or venv
site-packages to run against development code instead of the staged
package.

Builds are produced per target triple; variants are suffixed
(`-freethreaded`, `-jit`):

```bash
python scripts/build_native.py --run                  # host, deterministic pin
python scripts/build_native.py --python 3.14 --run    # pick a CPython
python scripts/build_native.py --python 3.13 --free-threaded
```

---

## 2. Ten-minute C++ tour

```cpp
#include "nadata.hpp"   // native/wrappers/cpp — include dir: native/

int main() {
    nadata::Database db = nadata::Database::load("./my_project");

    // Symbols, attributes, calls — 1:1 with the Python facade.
    int64_t tom = db.get_symbol_id("tom");
    auto birthplace = db.get_attribute("tom", "birthplace");  // -> Value
    db.set_attribute("tom", "birthplace", nadata::Value::str("Lyon"));

    bool older = db.call_method(
        "jean", "older",
        nadata::Value::array({nadata::Value::instance(tom)})).as_bool();

    // Queries chain on the same handle (the ABI rebinds it in place).
    nadata::Query q = db.query("Human");
    q.where_str("h", "h.name == \"Tom\"");
    auto id = q.first_id();          // std::optional<int64_t> — nullopt if empty

    // ABI 2: the same operations keyed by entity id instead of symbol
    // names — the id IS the object reference.
    if (id) {
        auto name = db.get_attr_by_id(*id, "name");
        db.set_attr_by_id(*id, "name", nadata::Value::str("Tom"));
        auto methods = db.list_methods(*id);   // std::vector<std::string>
        (void) name; (void) methods;
    }

    // Errors are exceptions carrying the ABI status:
    try {
        db.get_symbol_id("nobody");
    } catch (const nadata::Error& e) {
        // e.status() == ND_ERR_NOTFOUND, e.what() has the message
    }
}
```

A raw C host does the same with `nd_load` / `nd_get_symbol_id` / ... and
explicit status checks; `native/tests/smoke.c` is a complete example.

---

## 3. The API surface

Every `nd_*` function maps 1:1 onto `nadata.api.Nadata` (the "Choice A"
facade). Grouped:

| Group | Functions |
|---|---|
| Lifecycle | `nd_load`, `nd_open`, `nd_open_folder`, `nd_close` |
| Introspection | `nd_struct_names`, `nd_function_names`, `nd_symbol_names`, `nd_summary`, `nd_count`, `nd_total_instance_count` |
| Instances | `nd_get_symbol_id`, `nd_get_attribute`, `nd_set_attribute`, `nd_create_instance`, `nd_call_function`, `nd_call_method` |
| Queries | `nd_query_new`, `nd_query_where_str`, `nd_query_where_cb`, `nd_query_order_by`, `nd_query_limit`, `nd_query_count`, `nd_query_exists`, `nd_query_first_id`, `nd_query_last_id`, `nd_query_pluck`, `nd_query_sum_of/min_of/max_of/average_of`, `nd_query_to_list_ids`, `nd_query_free` |
| Persistence | `nd_dump`, `nd_save_changes`, `nd_compact`, `nd_set_loop_limit` |
| Execution | `nd_execute_schema`, `nd_execute_data`, `nd_execute`, `nd_execute_script_file` |
| Misc | `nd_runtime_info`, `nd_last_error`, `nd_free_value`, `nd_free_str` |

The language-neutral nadata-syntax filter `nd_query_where_str` is the
recommended predicate style for every binding (it is also the only one
available on WASM); `nd_query_where_cb` takes a native function pointer
and is for C/C++/Rust/JVM hosts.

### The by-entity-id API (ABI 2 / Choice B)

ABI 2 adds a parallel surface keyed by the 64-bit **entity id** instead
of symbol names — the same handle an `ND_INSTANCE` value carries. It is
append-only: every ABI 1 symbol is unchanged, and hosts feature-detect
with `nd_runtime_info`'s `"abi": 2`.

| Function | Contract |
|---|---|
| `nd_get_attr_by_id(db, id, name, &out)` | attribute read by id; unknown id/attribute → `ND_ERR_NOTFOUND` |
| `nd_set_attr_by_id(db, id, name, value)` | attribute write by id; `value` is a borrowed `NdValue` |
| `nd_call_method_by_id(db, id, name, args, kwargs, &out)` | method call by id; results/args follow the same rules as `nd_call_function` |
| `nd_list_attr(db, id, &out)` / `nd_list_methods(db, id, &out)` | `ND_ARRAY` of `ND_STR` — the struct's attributes / methods |
| `nd_instance_id_valid(db, id)` | 1 when the id resolves to a live instance, 0 otherwise (never raises) |
| `nd_get_by_nid(db, struct_name, nid, &out)` | resolve a struct's **nid** (natural id) to an `ND_INSTANCE`; unknown → `ND_ERR_NOTFOUND` |

The entity id is the stable object reference for host bindings: hold
ids from `nd_get_symbol_id` / `nd_query_*_id` / `nd_get_by_nid` and
operate on them directly. Invalid ids raise `ND_ERR_NOTFOUND`
(`KeyError` underneath), so unknown-handle handling is uniform across
the API.

---

## 4. Status codes and errors

Every fallible call returns `NdStatusCode`; `ND_OK` (0) means success,
negatives are error **classes** mapped 1:1 from the engine's Python
exceptions. The human-readable message for the most recent call on the
**current thread** is `nd_last_error(db)` (NULL when the last call
succeeded; valid until the next `nd_*` call — copy it if you keep it).

| Status | Python exception | Typical cause |
|---|---|---|
| `ND_ERR_NOTFOUND` (-1) | `KeyError` / `AttributeError` / `IndexError` | unknown struct, symbol, attribute; `order_by("no_such")` |
| `ND_ERR_NAME` (-2) | `NameError` | unresolvable identifier in executed source |
| `ND_ERR_VALUE` (-3) | `ValueError` | missing file, invalid `.ndb`, `dump` to existing file, `set_loop_limit(0)` |
| `ND_ERR_TYPE` (-4) | `TypeError` | argument/attribute mismatch; home-type results (below) |
| `ND_ERR_SYNTAX` (-5) | `SyntaxError` | invalid predicate/schema text |
| `ND_ERR_IO` (-6) | `OSError` | filesystem failures |
| `ND_ERR_LIMIT` (-7) | `LoopLimitError` | per-loop fuel cap exceeded |
| `ND_ERR_PERMISSION` (-8) | `PermissionError` | sandboxed I/O traversal |
| `ND_ERR_RUNTIME` (-9) | `RuntimeError` family | engine failure at run time |
| `ND_ERR_INVALID_ARG` (-10) | — | NULL handle/required string, bad `NdValue` tag (shim-level) |
| `ND_ERR_STATE` (-11) | — | handle used after free (shim-level) |
| `ND_ERR_MEMORY` (-12) | `MemoryError` | allocation failure inside the shim |

The mapping walks the exception MRO, so subclasses resolve to the most
precise entry (`PermissionError` beats `OSError`).

### The v1 home-type limitation

`DateTime` and `Color` results (e.g. `tom.birthday`, functions like
`calculate_age_difference`) are **not representable** in ABI v1: such
calls return `ND_ERR_TYPE` with a message naming the type. Keep such
logic in nadata-language scripts (`#manual` scripts, defaults) or wrap
it in engine functions that return representable values. Length-paired
strings and richer value variants are reserved for ABI 0.2.

---

## 5. Values (`NdValue`)

```c
struct NdValue {
    NdType tag;          // ND_NULL, ND_BOOL, ND_INT64, ND_FLOAT64,
                         // ND_STR, ND_ARRAY, ND_DICT, ND_INSTANCE
    union {
        int32_t boolean;
        int64_t i64;
        double f64;
        char* str;
        struct { NdValue* items; int64_t count; } array;
        struct { char** keys; NdValue* values; int64_t count; } dict;
        int64_t instance_id;
    } as;
};
```

- **Strings** are UTF-8, NUL-terminated. Embedded NUL bytes are not
  representable in this ABI version.
- **ND_INSTANCE** carries a stable 64-bit **entity id** (0-based;
  `get_symbol_id("tom")` returns one, `nd_query_to_list_ids` returns
  arrays of them). Passing `ND_INSTANCE` *into* a call resolves the
  live instance automatically — you never handle Python objects.
- **NULL args**: `nd_call_function(db, "f", NULL, NULL, &out)` means no
  arguments. `args` is an `ND_ARRAY`; `kwargs` an `ND_DICT` with string
  keys.

---

## 6. Memory ownership

- **Inputs are borrowed.** `const NdValue*` parameters are copied by
  the shim during the call; free/reuse them immediately after.
- **Outputs are owned by the caller.** `NdValue*` out-params and
  returned `char*` must be released with `nd_free_value` (recursive:
  frees nested strings, arrays, dicts) and `nd_free_str`.
- If you *build* values to pass in, allocate nested strings/arrays with
  `malloc` — the same allocator `nd_free_value` uses (the C++ wrapper's
  `Value` does this for you).
- Handles: `nd_close(db)`, `nd_query_free(qs)`. Using a handle after
  release is undefined behavior.

## 7. Concurrency

- The shim owns **all** CPython/GIL handling: call from any thread, in
  any host configuration (including hosts that already embed Python).
- A database is **externally serialized**: concurrent calls are safe
  but execute one at a time. Use separate databases for parallelism.
- `nd_query_where_cb` callbacks run on the calling thread with the GIL
  attached and **must not call back into the same db handle**
  (deadlock). Keep callbacks pure and fast; prefer `where_str` when
  the predicate can be expressed in the nadata language.

## 8. Queries — the rebind-in-place contract

Python's `QuerySet` methods return *new* objects; the ABI keeps **one
handle** and rebinds it internally, so filters chain on the same
pointer:

```c
nd_query_new(db, "Human", &qs);
nd_query_where_str(qs, "h", "h.age(DateTime.now()).year >= 18");
nd_query_order_by(qs, "name", 0);
nd_query_limit(qs, 10);
nd_query_to_list_ids(qs, &out);   // ND_ARRAY of ND_INSTANCE
nd_query_free(qs);
```

- An **empty set is not silent**: `nd_query_first_id` /
  `nd_query_last_id` return `ND_ERR_NOTFOUND` (explicit, catchable) —
  ids are 0-based, so no sentinel value would be safe. Count/exists/
  to_list work fine on empty sets.
- `nd_query_limit(n)` with `n <= 0` is `ND_ERR_INVALID_ARG`
  (validated by the shim, before the engine is involved).
- `where_str` parses with the engine's own expression parser: syntax
  errors → `ND_ERR_SYNTAX`, evaluation type errors → `ND_ERR_TYPE`.
  String literals use double quotes only.

## 9. Persistence and execution

| Call | Contract |
|---|---|
| `nd_dump(db, path)` | full snapshot; **the file must not exist** (`ND_ERR_VALUE` otherwise) |
| `nd_save_changes(db, path, &docs)` | incremental delta append; missing file → `ND_ERR_VALUE` (dump first); `*docs` = documents appended |
| `nd_compact(db, NULL, &before, &after)` | rewrite to live state, in place; or pass `output_path`; returns document counts |
| `nd_set_loop_limit(db, n)` | raises the per-loop fuel cap; `n < 1` → `ND_ERR_VALUE`; persisted in db settings |
| `nd_execute_schema` / `nd_execute_data` / `nd_execute` | merge schema / create+bind instances / auto-detect — all against the live db |
| `nd_execute_script_file(db, path, params)` | runs a `#manual` script with its parameter prelude; `params` is an `ND_DICT` or NULL |

`nd_runtime_info()` returns JSON such as
`{"abi":2,"cpython":"3.12.7","free_threaded":false,"build":"embedded-pbs"}`
— use it to detect the bundled interpreter, build variant, and ABI
level.

---

## 10. Other languages

The header is deliberately bindgen/JNA-friendly (plain C, default
calling convention, no `__stdcall`, no struct return values):

- **C++**: `native/wrappers/cpp/nadata.hpp` (reference binding).
- **Java**: `native/wrappers/java/` — JNA 5.14 binding (JDK 11+),
  72/72 conformance cases green. `NdValue` params are passed as raw
  32-byte `Memory` blocks (tag @ 0, payload @ 8) — simpler and faster
  than `Structure`/`Union`. `nd_query_where_cb` is intentionally not
  mapped (JNA callbacks cannot honour the GIL contract); use
  `whereStr`. Run: `python native/wrappers/tests/java/run.py` —
  add `--bench` for the benchmark port.
- **Python hosts** (ctypes): load the dll, set `argtypes`/`restype`,
  build `NdValue` via `struct.pack` — see `native/tests/repro_call_*.py`.
- **Kotlin**: `native/wrappers/kotlin/` — thin idiomatic layer over
  the Java binding (operators, properties, `Long?` optionals,
  `getOrNull`, AutoCloseable `using {}`); 76/76 conformance cases
  green. No separate benchmark: extensions compile to the same Java
  calls. Run: `python native/wrappers/tests/kotlin/run.py`.
- **Kotlin/Java (raw)**: reuses the Java JNA binding directly, or
  Panama `java.lang.foreign`.
- **Rust**: `bindgen` over `nadata.h`, then a safe wrapper.
- **JavaScript**: Pyodide-style embedding; only `where_str` filters.

Conformance: `tests/test_abi_conformance.py` is the canonical case-id
registry; every binding replays the same ids plus the benchmark
protocol (see `native/wrappers/tests/README.md`).

## 11. ABI stability rules

- Symbols are never removed; signatures never change. New symbols are
  appended (the ABI 2 by-entity-id API did exactly this), new enum
  values get new numeric slots.
- `ND_ABI_VERSION` bumps when symbols are added — hosts feature-detect
  with it.
- The shim must never require host environment variables; the staged
  layout is located relative to the library itself.
