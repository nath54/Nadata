# Nadata Native ABI — Design Thinking

Planning document for exposing Nadata natively to other languages
(C/C++, Kotlin, Java, Rust, JavaScript) via a compiled `.dll`/`.so`
library. **No code yet** — this is the Phase 0 design artifact.

---

## Part 0 — Feasibility framing

Nadata is a pure-Python interpreter engine; it needs a runtime. The
realistic path is **not** "compile Python away" but:

> **Embed CPython + expose a stable C ABI shim over the Python facade.**

One C ABI unlocks everything downstream:

| Target | Mechanism | Effort |
|---|---|---|
| C/C++ | link the `.dll`/`.so` + header | trivial |
| Java/Kotlin | JNA (or Panama `java.lang.foreign` on modern JDKs) — calls the C ABI directly, no JNI code | small |
| C#/.NET | P/Invoke | small |
| Rust | `bindgen` over `nadata.h` | small |
| Swift/Obj-C | direct C interop | small |
| Web JS | **Pyodide** (CPython→WASM, `micropip install nadata` works because zero-dependency) or a WASM build of the shim | small–medium |

Rejected alternatives:

- **Nuitka/Cython → "true" standalone .dll**: still requires a CPython
  runtime to host the objects; you end up at the embed approach anyway.
- **Rewriting the engine in C++/Rust** or transpiling `.nastruct` →
  native code: a different multi-year project.

Why Nadata is unusually well positioned:

- `nadata/api.py` is already a clean facade taking/returning **plain
  Python values** (`_wrap_argument` / `_unwrap_value`) — the marshalling
  layer maps 1:1.
- **Zero dependencies** — no vendoring nightmare when bundling the
  interpreter.
- Cython-compiling hot paths (tokenizer, evaluator, indexes) *inside
  the embedded runtime* gives native-feeling speedups for free.

---

## Part 1 — The actual Python API (ground truth)

Everything hangs off one facade class, `nadata.Nadata`. Boundary rule:
**arguments in = plain Python values, results out = plain Python
values** (`int`, `float`, `str`, `bool`, `list`, `dict`,
`StructInstance`).

### 1.1 Lifecycle

| Python | Inputs | Returns |
|---|---|---|
| `Nadata.load(path)` | `str` — folder, `.nastruct`, or `.ndb` (auto-detected) | `Nadata` |
| `Nadata.open(path)` | `str` — existing `.ndb` | `Nadata` |
| `Nadata.open_folder(path, db_path=None, run_auto=True, force_init=False)` | session folder + options | `Nadata` |
| `dump(storage_path)` | `str` | `None` (full snapshot) |
| `save_changes(storage_path)` | `str` | `int` (delta docs appended) |
| `compact(output_path=None)` | `str?` | `tuple[int, int]` |

### 1.2 Introspection

| Python | Returns |
|---|---|
| `struct_names` / `function_names` / `symbol_names` | `list[str]` |
| `count(struct_name)` | `int` — polymorphic (includes subclasses) |
| `total_instance_count()` | `int` |
| `summary()` | `str` |

### 1.3 Instances & attributes

| Python | Inputs → Returns |
|---|---|
| `get(symbol)` | `str` → `StructInstance` |
| `get_attribute(symbol, name)` | `str, str` → plain value |
| `set_attribute(symbol, name, value)` | `str, str, Any` → `None` |
| `create_instance(struct_name, attributes, symbol=None)` | `str, dict[str, Any], str?` → `StructInstance` |
| `call_method(symbol, name, *args, **kwargs)` | → plain value |
| `call_function(name, *args, **kwargs)` | → plain value |

### 1.4 Queries — `QuerySet` (fluent)

```python
qs = app.query("Human")                        # -> QuerySet
qs.where(lambda inst: ...)                     # -> QuerySet  (Python callable)
qs.order_by("age", descending=True)            # -> QuerySet
qs.limit(5)                                    # -> QuerySet
qs.count / qs.exists() / qs.first() / qs.last()
qs.pluck("name")                               # -> list[Any]
qs.sum_of("x") / min_of / max_of / average_of  # -> Any / float
qs.to_list()                                   # -> list[StructInstance]
```

### 1.5 Code execution

| Python | Inputs → Returns |
|---|---|
| `execute_schema(text)` | `.nastruct` source → `None` (live merge) |
| `execute_data(text)` | `.nadata` source → `None` |
| `execute(text)` | auto-detects schema vs data → `None` |
| `execute_script_file(path, params)` | `str, dict[str, Any]` → `None` |

### 1.6 Error model (the part FFI cares most about)

- `KeyError` — unknown struct
- `NameError` — unknown symbol / unresolvable identifier
- `ValueError` — bad `.ndb` file, `save_changes` on missing file
- `TypeError` — argument mismatch in method calls
- `SyntaxError` — resolve-time schema errors
- `LoopLimitError` — per-loop fuel cap exceeded
- `PermissionError` — sandboxed I/O traversal in session projects

**Every one of these must become a return code + message at the ABI
boundary — exceptions must never cross.**

FFI-hostile elements to design around:

1. `where(lambda ...)` — *callbacks*; the hardest thing to marshal
   (impossible in WASM).
2. `StructInstance` objects — must not leak as raw pointers to GC'd
   languages; use **stable entity ids** instead.
3. `*args/**kwargs` — needs explicit arg-list forms.
4. `Any` — needs a tagged-union value type.

---

## Part 2 — Architecture: one C ABI, five skins

```text
             nadata.h  (stable C ABI — the single source of truth)
   ┌────────┬────────┬─────────┬──────────┬─────────────┐
   C/C++    Kotlin    Java      Rust        JavaScript
   (link)   (JNA)     (JNA or   (bindgen)   (Pyodide or
                       Panama)               WASM shim)
```

### Core decisions

1. **Opaque handle** `NdDb*` — never expose Python objects.
2. **Instances as `int64_t` entity ids**, not pointers — safe across
   GC, serializable, survives compaction.
3. **`NdValue` tagged union** for all dynamic values — the FFI
   equivalent of "plain Python value".
4. **Every function returns `NdStatus`** (int32: 0 = ok, negative =
   error class), with `nd_last_error(db) -> const char*`. Mapping to
   each language's native error mechanism happens in the wrapper.

### The callback problem (crucial design choice)

`QuerySet.where` gets **two** bindings:

- `nd_query_where_str(qs, "h", "h.name == 'Tom'")` — a **string
  predicate in the nadata language**, evaluated by the engine. Works
  in *every* target language, including WASM. This is the primary,
  universal form.
- `nd_query_where_cb(qs, callback)` — a C function pointer
  `(int64 id, void* user_data) -> bool`. Usable from C/C++/Rust and
  JVM (JNA `Callback`). Unavailable in WASM.

---

## Part 3 — The C ABI (root of everything)

### Value representation

```text
NdType:: NULL | BOOL | INT64 | FLOAT64 | STR | ARRAY | DICT | INSTANCE

struct NdValue { tag; bool b; int64 i; double d; NdStr s; /* array/dict heap slots */ }
```

### Function families (name → signature shape)

```text
Lifecycle      nd_load(path) -> NdDb*          nd_open(path) -> NdDb*
               nd_open_folder(path, db_path, run_auto, force_init) -> NdDb*
               nd_close(db)
Introspection  nd_struct_names(db) -> NdValue  nd_count(db, struct) -> int64
               nd_total_instance_count(db) -> int64   nd_summary(db) -> NdStr
Instances      nd_get_symbol_id(db, symbol) -> int64
               nd_create_instance(db, struct, NdValue attrs, symbol?) -> int64
               nd_get_attr(db, id, name, out NdValue*) -> NdStatus
               nd_set_attr(db, id, name, NdValue) -> NdStatus
               nd_call_method(db, id, name, args: NdValue[], kwargs: (name, NdValue)[], out)
               nd_call_function(db, name, args, kwargs, out)
QuerySet       nd_query_new(db, struct) -> NdQuery*
               nd_query_where_str(qs, var, expr)   nd_query_where_cb(qs, fnptr, user_data)
               nd_query_order_by(qs, attr, descending)   nd_query_limit(qs, n)
               nd_query_count / first_id / exists / pluck / sum_of / average_of
Persistence    nd_dump(db, path)   nd_save_changes(db, path) -> int64
Execution      nd_execute_schema(db, text)   nd_execute_data(db, text)
               nd_execute_script_file(db, path, params: NdValue)
Memory/Errors  nd_free_value(NdValue*)  nd_free_str(char*)  nd_last_error(db) -> const char*
```

Ownership rule: **caller frees** every `NdValue`/string the API hands
back; caller owns every `NdValue` it passes in.

### C usage sketch

```c
NdDb* db = nd_load("examples/ex0_simple");
int64_t tom = nd_get_symbol_id(db, "tom");
NdValue v; nd_get_attr(db, tom, "birthplace", &v);   // v.tag == ND_STR
nd_free_value(&v);
int64_t n = nd_count(db, "Human");
nd_close(db);
```

---

## Part 4 — Per-language interfaces

*(Interface signatures only — specification, not implementation.)*

### 4.1 C++

RAII wrapper mapping `NdStatus` → exceptions and `NdValue` →
`std::variant`.

```cpp
namespace nadata {
  using Value = std::variant<std::monostate, bool, int64_t, double,
                             std::string, std::vector<Value>,
                             std::map<std::string, Value>, Instance>;

  class Instance { public: int64_t id() const;
                   Value get(const std::string& attr) const;
                   void set(const std::string& attr, Value v); };

  class QuerySet { public: QuerySet& where(const std::string& var, const std::string& expr);
                           QuerySet& where(std::function<bool(const Instance&)> pred);
                           QuerySet& order_by(const std::string& attr, bool descending = false);
                           QuerySet& limit(int64_t n);
                           int64_t count() const;  bool exists() const;
                           std::optional<Instance> first() const;
                           std::vector<Value> pluck(const std::string& attr) const; };

  class Database { public: static Database load(const std::string& path);
                   static Database open(const std::string& path);
                   static Database openFolder(const std::string& path, /* options */);
                   ~Database();                       // non-copyable, movable
                   std::vector<std::string> structNames() const;
                   int64_t count(const std::string& structName) const;
                   Instance get(const std::string& symbol) const;
                   Instance create(const std::string& structName,
                                   std::map<std::string, Value> attrs,
                                   const std::string& symbol = "");
                   Value attr(const std::string& symbol, const std::string& name) const;
                   void setAttr(const std::string& symbol, const std::string& name, Value v);
                   Value callFunction(const std::string& name, std::vector<Value> args);
                   Value callMethod(const std::string& symbol, const std::string& name,
                                    std::vector<Value> args);
                   QuerySet query(const std::string& structName) const;
                   void dump(const std::string& path) const;
                   int64_t saveChanges(const std::string& path) const;
                   void executeSchema(const std::string& text);
                   void executeData(const std::string& text); };
}
```

Errors: `nadata::NotFoundError : std::runtime_error`, `TypeError`,
`SyntaxError` — thrown by the wrapper when status < 0.

### 4.2 Kotlin

Two layers: a private JNA `Library` binding and the idiomatic public
interface. Kotlin users never see pointers.

```kotlin
// Public API — idiomatic Kotlin
sealed class NdValue {
    data class Bool(val v: Boolean) : NdValue()
    data class Int(val v: Long)     : NdValue()
    data class Float(val v: Double) : NdValue()
    data class Str(val v: String)   : NdValue()
    data class Arr(val v: List<NdValue>) : NdValue()
    data class Dict(val v: Map<String, NdValue>) : NdValue()
    data class Instance(val id: Long)   : NdValue()
}

interface NadataDb : AutoCloseable {
    companion object { fun load(path: String): NadataDb
                       fun open(path: String): NadataDb
                       fun openFolder(path: String, dbPath: String? = null,
                                      runAuto: Boolean = true, forceInit: Boolean = false): NadataDb }

    val structNames: List<String>
    fun count(structName: String): Long
    fun totalInstanceCount(): Long
    fun summary(): String
    fun get(symbol: String): NdValue.Instance
    fun getAttribute(symbol: String, name: String): NdValue
    fun setAttribute(symbol: String, name: String, value: NdValue)
    fun createInstance(structName: String, attributes: Map<String, NdValue>,
                       symbol: String? = null): NdValue.Instance
    fun callFunction(name: String, args: List<NdValue> = emptyList(),
                     kwargs: Map<String, NdValue> = emptyMap()): NdValue
    fun callMethod(symbol: String, name: String, args: List<NdValue> = emptyList(),
                   kwargs: Map<String, NdValue> = emptyMap()): NdValue
    fun query(structName: String): NdQuery
    fun dump(path: String); fun saveChanges(path: String): Long
    fun executeSchema(text: String); fun executeData(text: String)
}

interface NdQuery {
    fun where(varName: String, predicate: String): NdQuery          // engine-evaluated
    fun where(predicate: (NdValue.Instance) -> Boolean): NdQuery    // JVM callback (JNA Callback)
    fun orderBy(attribute: String, descending: Boolean = false): NdQuery
    fun limit(n: Long): NdQuery
    fun count(): Long; fun exists(): Boolean
    fun first(): NdValue.Instance?; fun last(): NdValue.Instance?
    fun pluck(attribute: String): List<NdValue>
    fun sumOf(attribute: String): NdValue; fun averageOf(attribute: String): Double
    fun toList(): List<NdValue.Instance>
}
```

**How it's used** (Android or server app):

```kotlin
NadataDb.load("worlds/john.ndb").use { db ->
    val adults = db.query("Human")
        .where("h", "h.age(DateTime.now()).year >= 18")
        .orderBy("name").limit(10).pluck("name")
}
```

Errors → `NadataException(msg)` with sealed subtypes per status class.
The JVM callback form of `where` works because JNA converts a Kotlin
lambda to a C function pointer; on Android prefer the string form
(fewer JNI pitfalls).

### 4.3 Java

Identical shape, Java conventions: records for values,
`AutoCloseable`, `Optional`, fluent style.

```java
public sealed interface NdValue permits NdBool, NdInt, NdFloat, NdStr, NdArr, NdDict, NdInstance {}
public record NdInstance(long id) implements NdValue {}

public interface NadataDb extends AutoCloseable {
    static NadataDb load(String path) { ... }         // + open, openFolder(...)
    List<String> structNames();
    long count(String structName);
    NdInstance get(String symbol);
    NdValue getAttribute(String symbol, String name);
    void setAttribute(String symbol, String name, NdValue value);
    NdInstance createInstance(String structName, Map<String, NdValue> attrs, String symbol);
    NdValue callFunction(String name, List<NdValue> args, Map<String, NdValue> kwargs);
    NdValue callMethod(String symbol, String name, List<NdValue> args, Map<String, NdValue> kwargs);
    NdQuery query(String structName);
    void dump(String path);
    long saveChanges(String path);
    void executeSchema(String text);
    void executeData(String text);
}

public interface NdQuery {
    NdQuery where(String varName, String predicate);                 // universal
    NdQuery where(Predicate<NdInstance> predicate);                  // java.util.function overload
    NdQuery orderBy(String attribute, boolean descending);
    NdQuery limit(long n);
    long count(); boolean exists(); Optional<NdInstance> first();
    List<NdValue> pluck(String attribute);
}
```

Binding tech: **JNA** for the easy path; **Panama
(`java.lang.foreign`)** for JDK 22+ zero-reflection FFI. Usage:
`try (NadataDb db = NadataDb.load("store.ndb")) { ... }` with
`NadataException` on failure.

### 4.4 Rust

A `-sys` crate (auto-generated via `bindgen` over `nadata.h`) plus a
safe wrapper crate `nadata-rs`:

```rust
// Safe wrapper — public API
pub enum NdValue {
    Null, Bool(bool), Int(i64), Float(f64), Str(String),
    Array(Vec<NdValue>), Dict(HashMap<String, NdValue>), Instance(i64),
}

pub struct Database { /* NonDrop guard + handle */ }

impl Database {
    pub fn load(path: impl AsRef<Path>) -> Result<Database, NdError>;
    pub fn open(path: impl AsRef<Path>) -> Result<Database, NdError>;
    pub fn open_folder(path: &Path, db_path: Option<&Path>,
                       run_auto: bool, force_init: bool) -> Result<Database, NdError>;
    pub fn struct_names(&self) -> Vec<String>;
    pub fn count(&self, struct_name: &str) -> Result<i64, NdError>;
    pub fn get(&self, symbol: &str) -> Result<Instance, NdError>;
    pub fn get_attribute(&self, symbol: &str, name: &str) -> Result<NdValue, NdError>;
    pub fn set_attribute(&self, symbol: &str, name: &str, value: NdValue) -> Result<(), NdError>;
    pub fn create_instance(&self, struct_name: &str, attrs: HashMap<String, NdValue>,
                           symbol: Option<&str>) -> Result<Instance, NdError>;
    pub fn call_function(&self, name: &str, args: &[NdValue]) -> Result<NdValue, NdError>;
    pub fn call_method(&self, symbol: &str, name: &str, args: &[NdValue]) -> Result<NdValue, NdError>;
    pub fn query(&self, struct_name: &str) -> Result<QuerySet<'_>, NdError>;
    pub fn dump(&self, path: &Path) -> Result<(), NdError>;
    pub fn save_changes(&self, path: &Path) -> Result<i64, NdError>;
    pub fn execute_schema(&self, text: &str) -> Result<(), NdError>;
    pub fn execute_data(&self, text: &str) -> Result<(), NdError>;
}

impl Instance { pub fn id(&self) -> i64;
                pub fn get(&self, attr: &str) -> Result<NdValue, NdError>;
                pub fn set(&self, attr: &str, v: NdValue) -> Result<(), NdError>; }

impl QuerySet<'_> {
    pub fn where_str(self, var: &str, expr: &str) -> Result<Self, NdError>;
    pub fn where_fn(self, f: impl Fn(&Instance) -> bool) -> Self;   // fn-pointer + Box closure
    pub fn order_by(self, attr: &str, descending: bool) -> Result<Self, NdError>;
    pub fn limit(self, n: i64) -> Self;
    pub fn count(&self) -> i64; pub fn exists(&self) -> bool;
    pub fn pluck(&self, attr: &str) -> Result<Vec<NdValue>, NdError>;
}
```

`Database`/`QuerySet` own their handles with `Send`-guarded `Drop` →
`nd_close`/`nd_query_free`. Errors:

```rust
NdError { kind: NdErrorKind /* NotFound, TypeMismatch, Syntax, Io, Runtime */,
          message: String }
```

Usage:

```rust
let db = Database::load("store.ndb")?;
let names = db.query("Human")?.where_str("h", "h.age >= 18")?.pluck("name")?;
```

### 4.5 JavaScript (web)

Two delivery paths, same JS interface:

- **Path A — Pyodide** (recommended first): CPython compiled to WASM;
  `micropip.install("nadata")` works because zero-dependency. The JS
  class is a thin async wrapper over the Python facade. JS callbacks
  for `where` work naturally (Pyodide can call JS functions).
- **Path B — WASM build of the C shim**: compile `nadata.c` + embedded
  interpreter to WASM; string predicates only. Smaller, more build
  work — Phase 5.

```js
// Same interface under both paths
class NdQuery {
  where(varNameOrFn, predicateText)   // ("h", "h.name == 'Tom'") or (jsFn)
  orderBy(attribute, descending = false)
  limit(n)
  count() / exists() / first() / last()
  pluck(attribute)                    // -> Array
  sumOf(attr) / averageOf(attr) / toList()
}

class Nadata {
  static async load(path)             // async: interpreter boot on first call
  static async open(path)
  static async openFolder(path, { dbPath, runAuto = true, forceInit = false })
  get structNames()                   // -> string[]
  count(structName)                   // -> number (int)
  get(symbol)                         // -> NdInstance { id }
  getAttribute(symbol, name)          // -> number | string | boolean | Array | Object
  setAttribute(symbol, name, value)
  createInstance(structName, attrs, symbol?)
  callFunction(name, ...args)
  callMethod(symbol, name, ...args)
  query(structName)                   // -> NdQuery
  async dump(path)                    // browser FS / IndexedDB on web
  async saveChanges(path)             // -> bigint
  executeSchema(text) / executeData(text) / execute(text)
}

// Usage (browser or Node):
const db = await Nadata.load("./examples/ex0_simple");
const names = db.query("Human").where("h", "h.birthplace == 'Paris'").pluck("name");
```

JS↔engine value mapping: JS `number` splits to Int64/Float64 the same
way JSON round-trips do; JS objects ↔ dicts; arrays ↔ `Array[T]`;
`NdInstance` is a small object wrapping the id. File I/O on the web
routes through the browser virtual FS (Pyodide MEMFS) or IndexedDB.

---

## Part 5 — Unified type-mapping table

| Engine/Python | C ABI | C++ | Kotlin | Java | Rust | JS |
|---|---|---|---|---|---|---|
| `None` | `ND_NULL` | `std::monostate` | `NdValue.Null` | `NdNull` | `NdValue::Null` | `null` |
| `bool` | `ND_BOOL` | `bool` | `NdValue.Bool` | `NdBool` | `Bool` | `boolean` |
| `int` | `ND_INT64` | `int64_t` | `NdValue.Int` | `NdInt` | `Int` | `number`/`bigint` |
| `float` | `ND_FLOAT64` | `double` | `NdValue.Float` | `NdFloat` | `Float` | `number` |
| `str` | `ND_STR` (UTF-8 + len) | `std::string` | `NdValue.Str` | `NdStr` | `String` | `string` |
| `list` | `ND_ARRAY` | `std::vector<Value>` | `NdValue.Arr` | `NdArr` | `Array(Vec)` | `Array` |
| `dict` | `ND_DICT` | `std::map<string, Value>` | `NdValue.Dict` | `NdDict` | `Dict(HashMap)` | `Object`/`Map` |
| `StructInstance` | `int64` id | `Instance` | `NdValue.Instance` | `NdInstance` | `Instance(i64)` | `NdInstance` |
| exceptions | `NdStatus` + msg | exceptions | `NadataException` | `NadataException` | `Result<_, NdError>` | thrown `NadataError` / rejected promise |

---

## Part 6 — Build & release roadmap

1. **Phase 0** — Freeze the ABI surface doc (`nadata.h` as spec) + an
   ABI conformance test suite in Python that exercises every call, so
   every binding is tested against the same contract.
2. **Phase 1** — `scripts/build_native.py`: CPython embed + shim →
   `dist/native/{windows,linux,macos}/nadata.{dll,so,dylib}` +
   `nadata.h`. Use **abi3/`Py_LIMITED_API`** on Windows so one DLL
   spans Python 3.10+.
3. **Phase 2** — C++ wrapper (header-only) + a C smoke test in CI.
4. **Phase 3** — Kotlin/Java JNA wrapper (publish to Maven Central as
   `io.nadata:nadata`), JS Pyodide bundle (npm package `nadata-js`
   that lazy-loads Pyodide + the wheel).
5. **Phase 4** — Rust crate (`nadata-sys` + `nadata`), crates.io.
6. **Phase 5** — Cython-compile hot paths so native consumers get real
   speedups; optional pure-WASM path B.

### Biggest risks

- **Interpreter bundling** size/strategy — solved decision below
  (Part 8).
- **String predicates** must stay the *primary* documented filter
  style so every binding stays feature-complete.

---

## Part 7 — Concurrency contract (the GIL)

### What the engine does today

- `QueryExecutor` splits candidate sets into **chunks of 1024** and
  runs them on a **`ThreadPoolExecutor`** (default workers = CPU
  count). Small sets (≤ 1024) skip the pool entirely — no thread
  overhead for small queries.
- Results are concatenated in chunk order → **deterministic
  ordering**, parallelism is an implementation detail.
- `QueryOptimizer` loop-invariant hoisting (compute
  `target.age(now)` once, not N times) is often the bigger win.

### The honest GIL reality

The predicate path is pure Python. Under CPython's GIL, only one
thread runs Python bytecode at a time, so for **CPU-bound predicates**
the pool is *concurrency*, not *parallelism* (it can even be slightly
slower from context-switching). Where the pool **does** genuinely help
today: **I/O-bound work** — lazy `.ndb` row reads, decode-cache
misses, I/O builtins — the GIL is released during file I/O, so workers
overlap disk waits while others compute.

Read the current design as: *the parallel scaffolding is already
there; it becomes true parallelism the moment the runtime allows it.*

### Does it survive embedding? Yes — with three rules

1. **The GIL travels with the embedded interpreter.** It is a
   property of CPython, not of the host language. Engine behavior is
   identical standalone and embedded. Nothing is lost.
2. **Foreign threads must attach to the GIL.** When a Kotlin/Rust/C++
   thread calls into the library (including `where` callbacks fired
   from JNA worker threads), the C shim **must** wrap the call in
   `PyGILState_Ensure()` / `PyGILState_Release()`. **ABI rule: the
   shim is the only place that touches the GIL; native callers never
   do.** Missing this = crashes or deadlocks, not subtle bugs.
3. **The database is externally serialized.** Concurrent calls from
   the host are *safe* but execute one at a time on the GIL. The
   wrapper layer documents this; the Kotlin/Java wrapper may offer a
   single dedicated engine thread per database (actor model) for
   high-concurrency hosts. Rust marks the handle `!Send` initially.
   Pyodide/WASM is single-threaded, period (the ≤1k fast path keeps
   that sane).

### Upgrade path to true parallelism (no engine changes needed)

| Option | Effect | Cost |
|---|---|---|
| **Free-threaded CPython 3.13+** (PEP 703, no-GIL build) | the existing `ThreadPoolExecutor` gives real CPU parallelism with zero engine changes | experimental on Windows; needs testing |
| **Cython hot paths `nogil`** | compiled evaluator/index sections release the GIL → true parallel chunks today | Phase 5 work; helps only compiled sections |
| **Sub-interpreters** (PEP 684, per-interpreter GIL, 3.12+) | many DBs in one process, truly parallel | big complexity; for servers hosting many worlds |
| **Process pool** | true parallelism today, but instances/ids don't cross processes | breaks the object-graph model; rejected |

---

## Part 8 — Embedded interpreter distribution (distro-invariant)

Goal: `nadata.dll` / `libnadata.so` that works **anywhere**, with no
dependency on the user's installed Python — and Linux users can also
**compile it themselves**.

### Strategy: python-build-standalone (PBS) as the single runtime source

Use [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
(Astral-maintained, PSF-licensed, the same runtimes `uv` uses) as the
**one** source of interpreter binaries for all targets:

| Target | PBS build | Distro invariance |
|---|---|---|
| `x86_64-pc-windows-msvc` | windows build | self-contained; nothing from the registry |
| `x86_64-unknown-linux-gnu` | glibc build targeting **old glibc (2.17+)** | runs on Ubuntu/Debian/Fedora/Arch/RHEL/openSUSE — any modern glibc distro |
| `x86_64-unknown-linux-musl` | musl build | Alpine / fully-static-minded systems |
| `aarch64-unknown-linux-gnu` / `-musl` | arm64 builds | ARM servers, Raspberry Pi 5-class |
| `aarch64-apple-darwin` / `x86_64-apple-darwin` | macOS builds | Apple Silicon + Intel |

The build script **pins one PBS release, downloads it with a
checksum**, and never touches the system Python.

### Linking: static libpython = the distro-invariance trick

- **Linux**: link the **static** `libpython3.12.a` from PBS **into
  `libnadata.so` itself**. The interpreter lives inside our library →
  no `libpython3.x.so` dependency, no `LD_LIBRARY_PATH`, no distro
  Python version mismatch. This is what makes the `.so` work on any
  glibc ≥ 2.17 system.
- **Windows**: link the PBS import library; ship `python312.dll`
  alongside `nadata.dll` (or use the embeddable distribution).
- **macOS**: ship `libpython3.12.dylib` alongside (or static), then
  codesign + notarize the bundle.

### Runtime layout (what ships in `dist/native/<target>/`)

```text
libnadata.so          ← interpreter embedded inside
nadata.h
python312.zip         ← frozen-able stdlib (PBS provides it)
lib-dynload/          ← extension modules that must stay .so (zlib, sqlite3, ...)
nadata/               ← the nadata package itself
```

The shim **sets `PYTHONHOME` programmatically** (`PyConfig.home`) at
startup, relative to the library location — no environment variables,
works from any install path. The `sys.path` points at the bundled
stdlib + the embedded `nadata/` package.

### Compiling from source on Linux (for people who want it)

`scripts/build_native.py --from-source` drives the whole thing:

1. Download + build a pinned CPython from source
   (`./configure --prefix=... --with-static-libpython; make -j`) —
   works on any distro with a C toolchain; or use an installed
   `python3` if it matches the minimum version.
2. Compile the C shim against it (plain `cc` or CMake; no exotic
   dependencies — the shim is one file).
3. Stage the runtime layout above into `dist/native/`.

So both paths exist: **prebuilt per-target artifacts from CI** (the
GitHub Actions matrix uploads `dist/native/<target>.tar.zst` per
platform) **and** a from-source build for unusual distros or
off-line/air-gapped use.

### ABI doc additions required by this decision

- Document `nd_runtime_info()` → bundled CPython version, PBS tag,
  free-threaded flag (so hosts can detect the 3.13t case).
- Document the **no-env-var rule**: the shim must never depend on
  `PYTHONPATH`/`PYTHONHOME` set by the host environment.
- Size budget note: static-embedded runtime + stdlib ≈ 15–30 MB
  compressed per target — acceptable for a database engine, and the
  price of true distro invariance.

---

## Next step

Draft the actual `nadata.h` ABI spec (still no implementation) as the
Phase 0 artifact.
