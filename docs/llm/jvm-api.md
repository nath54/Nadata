# JVM API Reference (Java + Kotlin) — ABI 2

The complete public surface of the nadata JVM binding. Everything here
is code-free knowledge; signatures are exact as of v0.1.0
(`nadata.Nadata` in the fat jar or `native/wrappers/java/`).

## 0. Bootstrap

```java
// Fat jar (self-contained — the only classpath entry needed):
nadata.bundled.NadataBundled.install();   // extracts to ~/.nadata/native/<version>/

// Or an explicit library path (dev / staged bundle):
nadata.Nadata.loadLibrary("/path/to/nadata.dll");  // .dll/.so/.dylib
```

```kotlin
// Kotlin layer (NadataExt.kt):
import nadata.kotlin.loadLibrary
loadLibrary("/path/to/nadata.dll")        // or NadataBundled.install() first
```

`Nadata.Database.runtimeInfo()` (static) returns JSON:
`{"abi":2,"cpython":"3.12.7","free_threaded":false,"build":"embedded-pbs"}`.

## 1. `Nadata.Database` — implements `AutoCloseable`

Static factories:

| Method | Behaviour |
|---|---|
| `Database.load(String projectPath)` | Opens a **project folder** (`structs/` + `data/`) or a `.ndb`. The standard entry point. |
| `Database.open(String storagePath)` | Opens an existing `.ndb` dump (fails `ND_ERR_VALUE` if missing — use `load` for folders). |
| `Database.openFolder(String folderPath)` | Explicit folder form of `load`. |

Introspection:

| Method | Returns | Notes |
|---|---|---|
| `getStructNames()` | `List<String>` | All declared structs (incl. base classes). |
| `getFunctionNames()` | `List<String>` | Top-level `fn` functions. |
| `getSymbolNames()` | `List<String>` | Named instances from the data scripts. |
| `getSummary()` | `String` | Human-readable project summary. |
| `getCount(String structName)` | `long` | Polymorphic instance count (`"Entity"` counts subclasses). `ND_ERR_NOTFOUND` for unknown structs. |
| `getTotalInstanceCount()` | `long` | Every instance in the db. |
| `getSymbolId(String symbol)` | `long` | Entity id of a named instance. `ND_ERR_NOTFOUND` if unknown. |

Attribute access (by symbol name):

| Method | Notes |
|---|---|
| `Object getAttribute(String symbol, String name)` | Reads any attribute. `ND_ERR_NOTFOUND` for unknown symbol/attribute; `ND_ERR_TYPE` for values that cannot cross ABI 2 (DateTime, Color). |
| `void setAttribute(String symbol, String name, Object value)` | Idempotent writes allowed. Wrong type for the declared attribute → `ND_ERR_TYPE`; unknown symbol → `ND_ERR_NOTFOUND`. |

Creation, calls:

| Method | Notes |
|---|---|
| `long createInstance(String structName, Map<String,Object> attributes, String symbol)` | `attributes` may be `null` (defaults apply). `symbol` (optional name) may be `null`. Returns the new entity id. Unknown struct → `ND_ERR_NOTFOUND`. |
| `Object callFunction(String name, Object... args)` | Calls a top-level `fn`. Unknown → `ND_ERR_NOTFOUND`; bad arg types → `ND_ERR_TYPE`. |
| `Object callFunction(String name, Object[] args, Map<String,Object> kwargs)` | Explicit kwargs form. |
| `Object callMethod(String symbol, String name, Object... args)` | Calls `@name` on an instance (by symbol). Unknown method → `ND_ERR_NOTFOUND`. |

Persistence:

| Method | Notes |
|---|---|
| `void dump(String storagePath)` | Writes the **complete** db to a new `.ndb`. Fails `ND_ERR_VALUE` if the file exists. |
| `long saveChanges(String storagePath)` | Appends an **incremental delta** (only instances changed since load) to an existing `.ndb`. Returns docs written (≥ 1). Fails `ND_ERR_VALUE` if the target is missing. |
| `long[] compact(String outputPath)` | Rewrites the store without dead rows; `null` = in place. Returns `{before, after}` doc counts. |
| `void setLoopLimit(long limit)` | Per-loop fuel cap for script execution (guards infinite loops). |

Script/execution (nadata language — see the syntax reference):

| Method | Notes |
|---|---|
| `void executeSchema(String schemaText)` | Merges struct declarations into the live db (`class Weapon { ... }`). |
| `void executeData(String dataText)` | Executes data assignments/queries in-session. |
| `void execute(String text)` | Generic text (auto-detected). |
| `void executeScriptFile(String filePath, Map<String,Object> params)` | Runs a `.nadata` script file; `params` (nullable) become available variables. |

By-entity-id surface (ABI 2 — the stable object handle):

| Method | Notes |
|---|---|
| `Object getAttrById(long id, String name)` | Same semantics as `getAttribute`, keyed by entity id. Unknown id → `ND_ERR_NOTFOUND`. |
| `void setAttrById(long id, String name, Object value)` | |
| `Object callMethodById(long id, String name, Object... args)` | |
| `List<String> listAttr(long id)` | Declared attribute names present on the instance. |
| `List<String> listMethods(long id)` | Callable `@methods`. |
| `boolean instanceIdValid(long id)` | False for unknown/freed ids (no exception). |
| `OptionalLong getByNid(String baseStruct, String nid)` | Looks up an instance by its `nid = "..."` binding (declared in the data script). Empty `OptionalLong` when absent. |

## 2. `Nadata.Query` — implements `AutoCloseable`

`db.query(String structName)` starts a query (`ND_ERR_NOTFOUND` for
unknown structs). **Chaining rebinds in place** — the same Java object
is returned, no new resources:

| Method | Returns | Notes |
|---|---|---|
| `whereStr(String var, String text)` | `Query` | Engine-evaluated filter in the nadata expression language: `whereStr("h", "h.name == \"Tom\" && h.birthplace == \"Paris\"")`. Syntax errors → `ND_ERR_SYNTAX`; type errors → `ND_ERR_TYPE`. **This is the only filter form on the JVM** (callback filters are not exposed). |
| `orderBy(String attribute, boolean descending)` | `Query` | `ND_ERR_NOTFOUND` for unknown attributes. |
| `limit(long n)` | `Query` | `n < 1` → `ND_ERR_INVALID_ARG`. |
| `count()` | `long` | |
| `exists()` | `boolean` | |
| `firstId()` / `lastId()` | `OptionalLong` | Empty on an empty result set (the `ND_ERR_NOTFOUND` case is folded into empty). |
| `pluck(String attribute)` | `List<Object>` | Raw payloads in result order. Unknown attribute → `ND_ERR_NOTFOUND`. |
| `sumOf` / `minOf` / `maxOf` / `averageOf(String attribute)` | `double` | Numeric declared attributes only; sum/min/max accept int or float results. Undeclared attribute → `ND_ERR_NOTFOUND`; declared non-numeric (e.g. `Str`) → `ND_ERR_TYPE`. |
| `toListIds()` | `List<Long>` | All matching ids in instance order — **the way to enumerate results**. |
| `close()` | | Always close (or `use`/`using`). Closed handle use → `ND_ERR_STATE`. |

## 3. Value mapping (ABI 2)

| Engine value | Java input | Java output |
|---|---|---|
| `Str` | `String` | `String` |
| `Int` family | `Long`, `Integer` | `Long` |
| `Float` family | `Double`, `Float` | `Double` |
| `Bool` | `Boolean` | `Boolean` |
| `Array[...]` | `List<Object>` | `List<Object>` |
| dict/kwargs | `Map<String, Object>` | `Map<String, Object>` |
| struct instance / entity ref | `Nadata.instance(long id)` (marker) | `Long` � the raw opaque entity id |
| `Null` | — | `null` |
| `DateTime`, `Color` | **not representable** — calls touching them throw `Nadata.Error` (`ND_ERR_TYPE`) | same |

Keep DateTime/Color logic inside nadata-language scripts (methods or
`executeData`) and pass primitives across the boundary.

**Entity ids are opaque 64-bit values.** `0` can be a valid id � never
treat it as null/invalid; use `instanceIdValid(id)`. Arrays of
references (`Array[Struct]` attributes) read back as `List<Long>` of
ids. Pass ids back in with `Nadata.instance(id)` (args, attributes,
relation writes).

## 4. Errors

All failures throw `nadata.Nadata.Error extends RuntimeException`:

- `error.status` — the numeric status code
- `getMessage()` — `"<call>: <engine last_error text>"`

| Status | Constant | Meaning | Typical cases |
|---|---|---|---|
| 0 | `ND_OK` | success | |
| -1 | `ND_ERR_NOTFOUND` | missing | unknown struct/symbol/attribute/function/id; undeclared aggregate attribute |
| -2 | `ND_ERR_NAME` | unbound identifier | names not resolvable in expressions |
| -3 | `ND_ERR_VALUE` | bad value | `dump` to existing file, `open` of missing file, `saveChanges` to missing file |
| -4 | `ND_ERR_TYPE` | type mismatch | wrong attribute type, DateTime/Color crossing the ABI, non-numeric aggregate |
| -5 | `ND_ERR_SYNTAX` | syntax error | malformed `whereStr` expression, bad schema text |
| -6 | `ND_ERR_IO` | filesystem | unreadable paths |
| -7 | `ND_ERR_LIMIT` | loop fuel | script loops exceeded `setLoopLimit` |
| -8 | `ND_ERR_PERMISSION` | sandbox | script I/O traversal violations |
| -9 | `ND_ERR_RUNTIME` | engine failure | anything unexpected at run time |
| -10 | `ND_ERR_INVALID_ARG` | bad host call | `limit(0)`, null handles |
| -11 | `ND_ERR_STATE` | stale handle | using a closed `Database`/`Query` |
| -12 | `ND_ERR_MEMORY` | allocation | |

Observed mapping note: "unknown X" cases (`Unknown symbol 'nobody'`,
`Unknown function 'no_such_fn'`, undeclared attributes) surface as
**NOTFOUND (-1)** in practice, not NAME (-2). The conformance suite
tests these exact statuses.

## 5. Threading & lifetime rules

- **One `Database` per thread.** The shim serializes access per db
  handle and owns the GIL per call; sharing a handle across threads is
  not supported. For parallelism, open one `Database` per worker.
- `Database` and `Query` are `AutoCloseable`. Close `Query` promptly
  (`try`/`use`/`using`); use-after-close is `ND_ERR_STATE`.
- Input values are borrowed for the duration of a call; output values
  are freed by the binding immediately after conversion (no manual
  memory management from Java/Kotlin).
- `Query` chaining mutates the query in place; a `Query` is bound to
  its `Database` — closing the db invalidates its queries.

## 6. Kotlin layer (`nadata.kotlin.NadataExt`)

One file (`native/wrappers/kotlin/nadata/NadataExt.kt`) — drop it into
your sources (or a published jar) next to the fat jar:

| Idiomatic form | Underlying call |
|---|---|
| `db[symbol, attr]` / `db[symbol, attr] = value` | `getAttribute` / `setAttribute` |
| `db[id, attr]` / `db[id, attr] = value` | `getAttrById` / `setAttrById` |
| `db.getOrNull(symbol, attr)` | NOTFOUND → `null`, other statuses rethrow |
| `db.getAttrByIdOrNull(id, attr)` | same, by id |
| `db.firstIdOrNull()` / `lastIdOrNull()` / `db.getByNidOrNull(struct, nid)` | `OptionalLong` → `Long?` |
| `db.structNames`, `db.functionNames`, `db.symbolNames`, `db.summary`, `db.totalInstances` | getters |
| `db.countOf(struct)`, `db.symbolId(symbol)` | `getCount`, `getSymbolId` |
| `instance(id)` | `Nadata.instance(id)` |
| `db.using { ... }` / `query.using { ... }` | AutoCloseable use |

## 7. Performance envelope (Win11 reference, ops/sec)

| Operation | C++ | Java/JNA |
|---|---|---|
| load (ex0) | 61 | 58 |
| get_attribute | 563,739 | 80,227 |
| set_attribute | 219,935 | 93,025 |
| query where+count | 21,382 | 15,947 |
| create_instance | 108,906 | 34,254 |
| call_function | 18,252 | 15,214 |

Coarse operations are within 5-30% of C++; fine-grained get/set pay a
JNA dispatch tax. Peak RSS ~131 MB (JVM baseline ~56 MB). Batch work
into queries/`toListIds` rather than per-row attribute round-trips
when it matters.
