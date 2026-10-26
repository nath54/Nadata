# Nadata v0.1.1 — Native ABI distributions

Patch release: **two data-integrity fixes in the engine package** (which
ships inside every native bundle) plus the new **filtered random
sampling** API. The C ABI is unchanged (still version 2, same 49
functions) — existing host code, headers, and bindings compile and run
without modification.

- **ABI version**: 2 (unchanged — no shim/wrapper/header changes)
- **Engine**: nadata @ 0.1.1 (pure Python, zero dependencies)
- **Runtime**: python-build-standalone `20241016`, CPython 3.12.7
- **Validation**: full 1034-test suite + 78-case C++ conformance +
  47-check staged smoke on the build platform, all green

## Downloads

| File | Platform | Size |
|---|---|---|
| `nadata-native-x86_64-pc-windows-msvc.tar.gz` | Windows x64 (MSVC) | ~41 MB |
| `nadata-native-x86_64-unknown-linux-gnu.tar.gz` | Linux x64 (glibc) | ~34 MB |
| `nadata-native-aarch64-apple-darwin.tar.gz` | macOS Apple Silicon | ~17 MB |
| `nadata-jvm-0.1.1.jar` | JVM (JNA self-extracting bundle) | ~43 MB |

Verify with `SHA256SUMS` (attached): `sha256sum -c SHA256SUMS`.

## What changed vs v0.1.0

### Fixed — eviction data loss on large tables

`StructTable.all()` enumerated the decoded-row cache, so any table with
more rows than the decode bound (default 100,000) silently dropped the
evicted remainder: a 150k-row store reloaded as exactly 100k rows.
`all()` (and everything built on it — polymorphic queries, counts,
full scans) now enumerates `all_local_ids()` and reads each row,
re-decoding on demand. Regression-tested at the eviction bound.

### Fixed — batch documents with 123-byte headers crashed loading

A binary batch payload starts with a varint of its header length.
When the header is exactly 123 bytes (e.g. TagRef batches with 512
rows and 5-digit max_local_id), that varint is `0x7B` — the byte `{` —
and the loader misclassified the batch as JSON, crashing with
`UnicodeDecodeError` (and would have silently dropped every following
batch of the table). Batch classification now parses the varint +
header structurally, and the loader checks batches before JSON
documents.

### New — filtered random sampling

Draw random matching rows without decoding the whole table:

```text
Question.sample(400);                                        // random rows
Question.sample_by("kind", "qcm", 400);                      // hash-index fast path
Question.sample_where((q: Question) -> q.difficulty < 3, 400);
```

- Probes random rows (decode + test) until the quota fills; sparse
  filters fall back to scanning the remaining rows, so results are
  always correct.
- `sample_by` reads the `index(Struct, "attr")` hash bucket directly,
  decoding only the drawn rows.
- Python side: `app.sample(name, count, predicate?)`,
  `StructTable.sample/sample_by`, `QuerySet.sample(count)`.
- Reference: 400-of-100k draws in ~14 ms probed vs ~500 ms full scan.

See docs/api.md ("Filtered sampling") and docs/syntax.md §7.6.1.

## Migration from v0.1.0

None. Same ABI, same archive layout, same fat-jar loader
(`nadata-native/<target>/`, self-extracting, no environment needed).
Replace the old artifact and relink nothing — the shim binary and
headers are identical; only the embedded engine package changed.

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

## Known limitations

- `DateTime`/`Color` values do not cross the C ABI (v2): such calls
  return `ND_ERR_TYPE` — keep that logic in nadata-language scripts or
  engine functions. (Length-paired strings and richer values are
  planned for a future ABI.)
- A database handle must be used from one thread at a time
  (calls are internally serialized; use separate handles to parallelize).
- macOS x86_64 (Intel) is not built; Apple Silicon only for now.
- Bulk phases that touch every row while a table sits near the 100k
  decoded-row bound thrash the eviction cache — import in chunks.

## Commit

Built from the 0.1.1 tree by the `Native ABI build (dist)` workflow:
smoke-validated against the staged layout on windows-latest,
ubuntu-22.04, ubuntu-24.04 and macos-14.
