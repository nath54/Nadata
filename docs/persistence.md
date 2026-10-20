# Persistence

How Nadata stores a database in a single `.ndb` file — the format, the
full-dump path, and the incremental append path.

---

## The file format

An `.ndb` file is a **slotted page file holding framed JSON documents**.

- **Pages**: fixed-size (64 KiB) pages managed by `PageManager`. Each page
  has a small header (including a 32-bit free-space offset) and holds
  variable-size slots. Pages chain to the next page when full.
- **Records**: `RecordStore` frames documents into records and assigns each a
  sequential **document id**. Records can span pages.
- **Documents**: each record payload is a JSON document with a `kind` field.

Because the file is just a sequence of documents, a database file is
effectively a **log**: loading replays documents in insertion order, and
incremental saves simply append more documents.

## Document kinds

| `kind` | Contents |
|---|---|
| `meta` | Format version and `next_type_id` — the loader takes the **latest** one |
| `struct` | A full struct definition: name, parent name, attributes (type, default), methods (real ASTs) |
| `function` | A standalone function (body stored as a real AST) |
| `instance` | `struct_name`, `entity_id`, and an attribute map (decoded values or references) |
| `symbols` | A map of symbol name → entity id |

Two encodings deserve mention:

- **References** between instances are stored as `{"__ref__": entity_id,
  "__struct_type__": "..."}` markers — so an object graph survives
  persistence without duplication.
- **ASTs** (method and function bodies, default expressions) are stored as
  structured nodes, *not* source text: a reloaded database re-executes
  original code without reparsing.

## Full dump: `dump_database`

`dump_database(database, storage_path)` writes a **complete snapshot**:

1. One `meta` document.
2. Struct definitions (in registration order, so type ids stay stable).
3. Function definitions.
4. All instances, with attributes encoded via `ObjectCodec`.
5. Symbol bindings.

The target file **must not exist** — snapshots are not appends. `Nadata.dump`
and `nadata dump` both use this path.

## Loading: `load_database`

`load_database(storage_path)` replays the documents **in insertion order**:

1. **Meta** — validate format, take the latest header.
2. **Struct shells** — register every struct under its stable type id.
3. **Struct contents** — decode parents, attributes, and methods. A struct
   may be defined by *several* documents (see incremental saves); later ones
   **merge** into the shell, replacing colliding fields.
4. **Functions** — decode into callable `StructMethod`-style objects.
5. **Instances** — decode attribute values, resolving `__ref__` markers to
   the already-created instances. A later document for an entity id
   **updates** the existing instance.
6. **Symbols** — merged, so later documents can add or re-bind symbols.

## Incremental saves: `append_database_changes`

`append_database_changes(database, storage_path)` extends an existing file
with **only what changed**. The current state is diffed against the file's
contents (the file is loaded into a reference database), and the delta is
appended:

- **New structs** → a full `struct` document.
- **Extended structs** → a delta `struct` document containing *only* the new
  attributes / methods; the loader merges it into the existing shell.
- **New or changed instances** → an `instance` document with a full attribute
  snapshot for that entity. Changes are detected by comparing raw values,
  with instance **references compared by entity id** (so re-pointing a
  `parents` array is detected, and unchanged references do not force a
  rewrite).
- **New or re-bound symbols** → a `symbols` document.

An unchanged database appends nothing but the `meta` header — calling
`save_changes` repeatedly is safe. Each append also rewrites `next_type_id`
via its meta document so later appends keep stable type ids.

### The replay model

Because loading replays in order, the file after an append contains, e.g.:

```text
[meta v1] [structs...] [instances...] [symbols...]     # original dump
[meta v1] [instance: new one] [instance: changed one] [symbols: new]  # delta
```

The loader applies the delta *over* the snapshot — later documents win for
instances and symbols, and merge for structs. There is no compaction step
yet: repeated appends grow the file, and a `dump` to a new file is the
current way to compact.

## Python surface

```python
app.dump("store.ndb")              # full snapshot
n = app.save_changes("store.ndb")  # append delta, returns document count
```

## CLI surface

```bash
nadata dump examples/ex0_simple store.ndb   # full snapshot
nadata info store.ndb                       # inspect a saved database
```

## Invariants

- Type ids and entity ids are **stable across saves** — an entity id always
  refers to the same instance, in the file and across sessions.
- Bodies are stored as **ASTs**, never source strings.
- The format version is 1; `load_database` refuses files without a valid
  meta document.
