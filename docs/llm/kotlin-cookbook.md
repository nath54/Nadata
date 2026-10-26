# Kotlin Cookbook — building an app on nadata

Task-oriented recipes. All snippets assume the bootstrap from the
[API reference](jvm-api.md) and use the idiomatic Kotlin layer
(`nadata.kotlin.NadataExt`):

```kotlin
import nadata.Nadata
import nadata.NadataBundledHolder  // not needed — see below
import nadata.bundled.NadataBundled
import nadata.kotlin.using
```

```kotlin
// One-time bootstrap (see docs/distribution.md for setup):
NadataBundled.install()                    // fat jar: self-contained
val db = Nadata.Database.load("./myapp")
```

The running example: a small **task manager** — `User`, `Task`, `Tag`.

## 1. Project layout (recommended)

A nadata **project** is a plain directory:

```
myapp/
  structs/
    schema.nastruct        # struct declarations (+ methods, fns)
  data/
    seed.nadata            # initial data ("#init" session)
```

Ship this directory with your app, or create it programmatically with
`executeSchema`/`executeData` (recipe 9). For compiled artifacts prefer
a **dump** (`.ndb`) — a single self-contained file opened with
`Nadata.Database.open(...)`.

## 2. Declaring the schema

```kotlin
db.executeSchema("""
    class Item {
        name: Str;
        done: Bool = false;

        @toggle() -> Bool {
            self.done = !self.done;
            return self.done;
        }
    }

    class User {
        name: Str;
        email: Str = "";
        tasks: Array[Item] = [];
    }

    class Task(Item) {
        due: Int32 = 0;            # epoch days — keep DateTime in scripts
        priority: Int32 = 1;

        @is_overdue(now: Int32) -> Bool {
            return (!self.done) && (self.due < now);
        }
    }

    class Tag {
        label: Str;
        items: Array[Item] = [];
    }
""")
```

Rules that matter (full reference: `docs/syntax.md`):

- Types: `Str`, `Bool`, `Int8/16/32/64`, `UInt...`, `Float16/32/64`,
  `UFloat16/32`, `Array[T]`, struct types, `DateTime`, `Color`.
- Defaults with `=`.
- Methods are `@name(args) -> RetType { ... }`; the instance is
  `self`; top-level functions are `fn name(...) -> RetType`.
- Inheritance `class Task(Item)` — `Item` counts polymorphically in
  queries and `getCount("Item")`.
- **Keep `DateTime`/`Color` logic inside the schema** — they cannot
  cross the ABI (recipe 10).

## 3. Seeding data (in-session)

```kotlin
db.executeData("""
    alice = User(nid = "alice", name = "Alice", email = "alice@example.org");
    bob   = User(nid = "bob",   name = "Bob");

    t1 = Task(nid = "t1", name = "Write docs",  done = false, due = 20000, priority = 2);
    t2 = Task(nid = "t2", name = "Ship release", done = true,  due = 19999, priority = 1);

    alice.tasks = [t1];
    bob.tasks = [t2];

    urgent = Tag(nid = "urgent", label = "urgent", items = [t1]);

    flush();
""")
```

- `nid = "..."` gives the instance a **stable external key** for
  `getByNid` (recipe 8).
- `flush()` persists the session to the project's store.

## 4. CRUD

```kotlin
// Create — attributes optional (defaults apply); symbol optional.
val id = db.createInstance("Task",
    mapOf("name" to "Fix login", "priority" to 3L), "fix_login")

// Read
val name = db["fix_login", "name"]            // "Fix login"  (operator get)
val byId = db[id, "name"]                     // same, by entity id
val missing = db.getOrNull("fix_login", "nope")   // null, no exception

// Update
db["fix_login", "priority"] = 2
db[id, "done"] = true

// Delete: engine-side via data script (no ABI delete in v0.1.0)
db.executeData("""fix_login = null; flush();""")
```

## 5. Queries

```kotlin
// Filter + count
val openTasks = db.query("Task").apply {
    whereStr("t", "t.done == false")
}
openTasks.count()                       // e.g. 1

// Chain (rebinds in place — same object)
val urgentOpen = db.query("Task").apply {
    whereStr("t", "t.done == false && t.priority >= 2")
    orderBy("priority", true)
    limit(10)
}
val ids = urgentOpen.toListIds()        // List<Long> — enumerate results

// Aggregates
val avgPrio = db.query("Task").averageOf("priority")
val maxPrio = db.query("Task").maxOf("priority")

// pluck: raw column
val names = db.query("Task").apply { orderBy("name", false) }
    .pluck("name")                      // List<Any?> of String

// Polymorphic counting
db.countOf("Item")                      // Task instances count too
```

Expression language highlights (`whereStr`): `==`, `!=`, `<`, `>=`,
`&&`, `||`, `!`, attribute chains `t.name`, literals `"str"`, `123`,
`true`. Malformed → `Nadata.Error` status `ND_ERR_SYNTAX`; wrong types
→ `ND_ERR_TYPE`.

## 6. Relations (struct references and arrays)

Relations are **just attributes** whose type is a struct or
`Array[Struct]` — cross them with `instance(id)`:

```kotlin
val aliceId = db.getByNidOrNull("User", "alice")!!   // stable key -> id

// Read a relation: returns List<Any?> of Long (raw entity ids)
val tasks = db[aliceId, "tasks"] as List<*>
val firstTaskId = tasks[0] as Long
db[firstTaskId, "name"]                                // "Write docs"

// Write a relation with ids you already hold
db[aliceId, "tasks"] = listOf(Nadata.instance(id))

// Walk relations inside the engine instead (faster, no round-trips):
val doneNames = db.query("Task").apply {
    whereStr("t", "t.done == true")
}.pluck("name")
```

## 7. Calling functions and methods

```kotlin
// Top-level fn from the schema
val ok = db.callFunction("is_adult", Nadata.instance(aliceId)) as Boolean

// Method on an instance (by symbol or by id)
db.callMethod("alice", "toggle")                  // @toggle
db.callMethodById(id, "is_overdue", 20001L)       // @is_overdue(now)

// Explicit kwargs form
db.callFunction("is_adult", arrayOf(Nadata.instance(aliceId)),
    mapOf("now" to 20000L))
```

Arguments/returns follow the value-mapping table; entity params are
passed as `Nadata.instance(id)` and entity results come back as raw
`Long` ids (opaque � `0` can be valid, check `instanceIdValid`).

## 8. Stable keys: `nid` bindings

```kotlin
// Declared in data: alice = User(nid = "alice", ...)
val aliceId = db.getByNidOrNull("User", "alice")    // Long?
if (aliceId == null) error("user missing")
db[aliceId, "email"] = "new@example.org"
```

`nid` is the right key for **app-level rows synced across dumps** —
entity ids are stable within a store, `nid` is stable across
re-creation.

## 9. Persistence

```kotlin
// Full dump (target must not exist)
db.dump("./store.ndb")

// Open a dump
Nadata.Database.open("./store.ndb").using { restored ->
    restored.countOf("Task")
}

// Incremental save: only what changed since load is appended
db["fix_login", "priority"] = 5
val docs = db.saveChanges("./store.ndb")      // docs >= 1

// Reclaim space / merge deltas
Nadata.Database.open("./store.ndb").using { stale ->
    stale.compact(null)                       // in place, returns [before, after]
}
```

## 10. The DateTime/Color rule

`DateTime` and `Color` values **cannot cross the ABI** (both directions
throw `Nadata.Error`/`ND_ERR_TYPE`). Keep them engine-side:

```kotlin
// BAD — birthday is a DateTime attribute:
// db["tom", "birthday"]                      // throws ND_ERR_TYPE

// GOOD — expose Int/Str views in the schema and read those:
db.executeSchema("""
    class Person {
        name: Str;
        birthday: DateTime = DateTime();
        @birth_year() -> Int64 { return self.birthday.y; }
    }
""")
val year = db.callMethodById(personId, "birth_year") as Long
```

## 11. Threads and app architecture

```kotlin
// One Database per thread — never share a handle across threads.
val db = ThreadLocal.withInitial { Nadata.Database.load("./myapp") }

// UI thread (read model)
db.get().query("Task").apply { orderBy("due", false) }.toListIds()

// Worker thread (own handle)
thread {
    db.get().executeData("""flush();""")
}
```

Recommended shape: a thin repository per aggregate
(`TaskRepository(db)`) that owns the `whereStr` strings, so the UI
never touches raw ids. Long-lived reads go through `toListIds()` +
`getAttrById` batches; short-lived filters can `pluck` directly.

## 12. Testing your data layer

The engine is fully in-memory per handle — seed a temp project per
test (write the two files, load, assert) exactly like the binding's
own conformance suite does. `dump` to a temp `.ndb` to test
persistence round-trips.
