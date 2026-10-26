# Gotchas, limits and error behaviour

Things that surprise even experienced users — read once before
writing real code. All statuses/constants are defined in
[jvm-api.md](jvm-api.md) §4.

## 1. DateTime and Color never cross the ABI

Any read/write/arg/result touching a `DateTime` or `Color` value
throws `Nadata.Error` (`ND_ERR_TYPE`). This is by design (ABI 2 has
no payload for them). Pattern: expose `Int`/`Str` views via `@methods`
in the schema and call those. See cookbook recipe 10.

## 2. Entity id 0 is a valid id

Ids returned by `createInstance`/`getSymbolId`/relation reads are
opaque 64-bit values. `0` can absolutely be a valid id � never use it
as a null sentinel. The only truth is `instanceIdValid(id)`.

## 3. "Unknown X" is NOTFOUND (-1), not NAME (-2)

The contract maps Python `NameError` → `ND_ERR_NAME`, but in practice
all of these surface as **NOTFOUND**:

- unknown symbol (`getSymbolId("nobody")`)
- unknown attribute (`getAttribute("tom", "no_such")`)
- unknown struct (`getCount("Nope")`, `query("Nope")`)
- unknown function (`callFunction("no_such_fn")`)
- unknown method (`callMethod("tom", "no_such_method")`)
- unknown entity id (`getAttrById(12345, "name")`)
- undeclared attribute in aggregates/pluck (`sumOf("nid")`)

`ND_ERR_NAME` appears for unbound identifiers inside **expressions**.
Handle `ND_ERR_NOTFOUND` as the general "missing" case; `getOrNull`
/`getAttrByIdOrNull`/`*IdOrNull` in the Kotlin layer already do.

## 4. Aggregates distinguish "undeclared" from "non-numeric"

- `sumOf("nid")` where `nid` is **not declared** on the struct →
  `ND_ERR_NOTFOUND`.
- `sumOf("label")` where `label: Str` **is declared** →
  `ND_ERR_TYPE`.
- Declared but all-null numeric attributes aggregate as `0`/`0.0`
  rather than failing.

## 5. dump / saveChanges / compact expectations

- `dump(path)` **fails if the file exists** (prevents clobbering) —
  delete first or use a unique name.
- `saveChanges(path)` **requires an existing** `.ndb` written by
  `dump`; it appends a delta and returns the doc count.
- `compact(null)` rewrites in place; the returned array is
  `{before, after}`.
- Opening a folder-style project of a dump: use `load` for both; use
  `open` only for existing `.ndb` files (missing file → `ND_ERR_VALUE`).

## 6. One Database per thread

The shim serializes each handle; sharing a `Database` (or its
`Query` objects) across threads is unsupported and can corrupt state.
One handle per worker thread is the supported parallelism model —
cheap (loads are ~17 ms).

## 7. Query chaining rebinds in place

`q.whereStr(...)`/`orderBy(...)`/`limit(...)` mutate and return the
same object. This is fine for chaining; do not expect immutable
snapshots. A `Query` is invalid once its `Database` closes
(`ND_ERR_STATE`).

## 8. No callback filters on the JVM

The C ABI has `nd_query_where_cb` (host-provided predicate), but the
JVM binding intentionally does not expose it (JNA callbacks cannot
honour the GIL contract). Use `whereStr` — the engine evaluates the
expression natively, which is also faster. Anything not expressible
in `whereStr` goes into a schema `@method` you call per result id.

## 9. Loop fuel

Scripts executed via `executeData`/`executeScriptFile` run under a
per-loop fuel cap (`setLoopLimit`, default engine value). Runaway
loops raise `ND_ERR_LIMIT` instead of hanging your app. Raise it
deliberately for heavy data scripts.

## 10. Sandbox

Script I/O is sandboxed — traversal outside the project root raises
`ND_ERR_PERMISSION`. Ship data files inside the project folder.

## 11. Performance shape

- Per-call overhead dominates fine-grained get/set on the JVM (~12 µs
  vs ~1.8 µs in C++). Batch: `pluck`, `toListIds` + `getAttrById`,
  or push logic into schema `@methods`.
- `load` is ~17 ms — per-thread handles are cheap.
- Peak RSS ~131 MB on the JVM (baseline ~56 MB is the JVM itself).
- Never filter large tables with full scans per request (~1 s per 100k
  rows, worse with set membership). Use `index(Struct, "attr")` +
  `get_by` for O(1) equality, and the **sampling family** —
  `Struct.sample(count)` / `sample_by(attr, value, count)` /
  `sample_where(predicate, count?)` — to draw random matches without
  decoding the whole table (see docs/api.md "Filtered sampling").
- Bulk phases that touch every row while the table sits near the
  100k decoded-row bound thrash the eviction cache (53 s vs 0.8 s for
  a cold index build over 100k rows). Import in chunks, or keep hot
  tables under the bound.

## 12. Platform notes

- Windows: the bundle ships `python312.dll` next to the shim so a
  single `PATH` entry (or the fat jar's absolute-path preload) works.
- Linux: `LD_LIBRARY_PATH` **outranks** embedded rpaths — sanitize the
  environment when spawning from exotic runners (the fat-jar loader
  needs no env at all).
- macOS: dylibs are ad-hoc signed; no Intel build exists (arm64 only).
- CPython is pinned at 3.12.7 (PBS embedded build); `runtime_info`
  reports it.

## 13. What does NOT exist in v0.1.0 (avoid inventing APIs)

- No delete-instance call on the ABI (use `executeData` with
  `name = null; flush();`).
- No schema migration/eviction API (re-dump a transformed copy).
- No async/reactive API — calls are synchronous; wrap in your own
  executors.
- No transactions/isolation across handles — one writer at a time
  (per-handle serialization is per-handle, not cross-handle file
  locking; coordinate app-side or compact between writers).
