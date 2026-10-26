# LLM Knowledge Pack — build apps with nadata without reading the code

This directory is a **self-contained briefing pack**: feed these files
(and the existing guides they link) to an LLM and it can write correct
applications against nadata — no code exploration required. Every file
here is prose-first and API-exact; nothing requires reading `nadata/`
or `native/src/` sources.

## Read order (for a JVM/Kotlin app developer)

| # | Document | Why |
|---|----------|-----|
| 1 | [jvm-api.md](jvm-api.md) | **The complete Java/Kotlin API reference** — every method, semantics, value mapping, errors, threading. The single most important file. |
| 2 | [kotlin-cookbook.md](kotlin-cookbook.md) | Task-oriented Kotlin recipes: schema, CRUD, queries, relations, persistence, threading patterns. |
| 3 | [gotchas.md](gotchas.md) | Limitations, error-code table, observed error cases, performance envelope. Read before writing real code. |
| 4 | [../syntax.md](../syntax.md) | The **nadata language** (schema/data/script). Required to write `.nastruct`/`.nadata` files and any `executeSchema`/`executeData`/script content. |
| 5 | [../tutorial.md](../tutorial.md) | Walkthrough of a first project — reinforces the language. |
| 6 | [../distribution.md](../distribution.md) | How to add nadata to a project (install scripts, fat jar, CMake). |
| 7 | [../persistence.md](../persistence.md) | The `.ndb` format, dumps vs incremental saves vs compaction. |
| 8 | [../api.md](../api.md) | Python API + `nadata` CLI (only if scripting/tooling the project from Python or the command line). |
| 9 | [../architecture.md](../architecture.md) | Optional: engine internals (never needed for app development). |

## Quick orientation

Nadata is a **declarative data-schema and object-graph engine**:

- You declare **structs** (`.nastruct`) with typed attributes, defaults,
  relations (`Array[Struct]`, direct struct refs), methods (`@name()`)
  and top-level functions (`fn name(...)`) in the nadata language.
- You instantiate data in **project scripts** (`.nadata`) — plain
  assignments with keyword args, auto-tracked as a session graph.
- The engine runs **embedded CPython** internally; hosts talk to it
  through a small C ABI (`nadata.h`) wrapped per language (C++, Java,
  Kotlin, ctypes). From the JVM side everything is
  `nadata.Nadata` + `nadata.bundled.NadataBundled` (+ the Kotlin layer
  `nadata.kotlin`).
- A **project folder** is `structs/*.nastruct` + `data/*.nadata`;
  a **dump** is a single `.ndb` file; **incremental saves** append
  only changes; **compaction** rewrites the file.

Minimal example (ex0, the reference fixture used across all docs):

```python
# structs/structs.nastruct — class Entity { name: Str; ... }
# data/data1.nadata — tom = Human(name = "Tom", birthplace = "Paris");
```

From Kotlin:

```kotlin
NadataBundled.install()
Nadata.Database.load("./examples/ex0_simple").using { db ->
    println(db["tom", "birthplace"])                    // Paris
    db.query("Human").apply { whereStr("h", "h.name == \"Tom\"") }
        .count()                                        // 1
}
```

## Versioning note

This pack targets **ABI 2** (by-entity-id functions included) shipped
in release v0.1.0: embedded CPython 3.12.7, shim `nadata.dll` /
`libnadata.so` / `libnadata.dylib`, Java/Kotlin via JNA 5.14, JVM 11+.
