# Nadata v0.1.0 — Native ABI distributions

First public native release: the full Nadata engine (declarative
schemas, typed object graph, polymorphic queries, nadata-language
scripts) compiled into a single shared library with an **embedded
CPython 3.12.7** runtime — no Python installation, no environment
variables, no dependencies. Callable from **C, C++, and any language
with a C FFI** (Rust, Java/Kotlin via JNA, ...).

- **ABI version**: 2 (facade-mapping surface + by-entity-id surface)
- **Engine**: nadata @ 6409c6f (pure Python, zero dependencies)
- **Runtime**: python-build-standalone `20241016`, CPython 3.12.7
- **Validation**: full 1008-test suite + 78-case C++ conformance +
  47-check staged smoke on every platform, all green

## Downloads

| File | Platform | Size |
|---|---|---|
| `nadata-native-x86_64-pc-windows-msvc.tar.gz` | Windows x64 (MSVC) | 41 MB |
| `nadata-native-x86_64-unknown-linux-gnu.tar.gz` | Linux x64 (glibc) | 34 MB |
| `nadata-native-aarch64-apple-darwin.tar.gz` | macOS Apple Silicon | 17 MB |

Verify with `SHA256SUMS` (attached): `sha256sum -c SHA256SUMS`.

GitHub also auto-attaches `Source code (zip/tar.gz)` for this tag.

## What's inside each archive

```
native/<target>/
    nadata.dll | libnadata.so | libnadata.dylib  the shim
    nadata.lib                                   import lib (Windows)
    nadata.h                                     the C ABI — the only header hosts need
    nadata.hpp                                   header-only C++17 wrapper (nadata::)
    nadata/                                      the engine package (pure Python)
    runtime/                                     the pinned CPython 3.12.7
```

## 60-second start (C++)

```cpp
#include "nadata.hpp"   // point the include path at the extracted archive

int main() {
    nadata::Database db = nadata::Database::load("./my_project");

    // Choice A — symbol names, 1:1 with the Python facade:
    auto birthplace = db.get_attribute("tom", "birthplace");

    // Choice B — entity ids as object handles:
    int64_t tom = db.get_symbol_id("tom");
    auto methods = db.list_methods(tom);

    // Queries chain on one handle:
    nadata::Query q = db.query("Human");
    q.where_str("h", "h.name == \"Tom\"");
    auto id = q.first_id();   // std::optional<int64_t>
}
```

Link against the shim; make sure the shim's directory (or the archive
root) is on the loader path:

- **Windows**: link `nadata.lib`; put the archive root on `PATH`
- **Linux**: `-L<dir> -lnadata` — the interpreter is inside the .so
- **macOS**: `-L<dir> -lnadata` + `-rpath <dir>` (libpython resolves
  via the rpath into `runtime/lib`)

Full guide: [docs/native.md](https://github.com/nath54/Nadata/blob/main/docs/native.md) —
memory ownership, concurrency, persistence contracts, per-language
notes, and the ABI stability rules.

## Reference performance (Windows 11, MSVC 2022)

| Workload | C++ (embedded) | Python facade | SQLite floor |
|---|---|---|---|
| get_attribute | 550k/s | 1.20M/s | 640k/s |
| set_attribute | 226k/s | 302k/s | 876k/s |
| query (parse+eval+count) | 22.4k/s | 22.8k/s | 664k/s |
| call_function | 18.9k/s | 20.8k/s | 582k/s |
| full project load | 61/s | 122/s | 1715/s |
| peak RSS | 25.6 MB | 35.0 MB | 35.6 MB |

The C++ and Python columns agree within noise on every engine path —
the C ABI adds no meaningful overhead. See `benchmarks/` in the repo
for the protocol and `benchmarks/decomposition.py` for the layer
breakdown.

## Known limitations

- `DateTime`/`Color` values do not cross the C ABI (v2): such calls
  return `ND_ERR_TYPE` — keep that logic in nadata-language scripts or
  engine functions. (Length-paired strings and richer values are
  planned for a future ABI.)
- A database handle must be used from one thread at a time
  (calls are internally serialized; use separate handles to parallelize).
- macOS x86_64 (Intel) is not built; Apple Silicon only for now.

## Commit

Built from `6409c6f` by the `Native ABI build (dist)` workflow:
smoke-validated against the staged layout on windows-latest,
ubuntu-22.04, ubuntu-24.04 and macos-14 (the macOS build includes the
dylib install-name fix verified by dyld trace).
