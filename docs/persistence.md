# Persistence: the `.ndb` format

How a Nadata database is written to disk, loaded back, appended to, and
compacted.

## The file format

An `.ndb` file is a **slotted page file holding framed documents** (format
version 2).

- **Pages**: fixed-size (64 KiB) pages managed by `PageManager`. Each page
  has a small header (including a 32-bit free-space offset) and holds
  variable-size slots. Pages chain to the next page when full.
- **Records**: `RecordStore` frames documents into records and assigns each a
  sequential **document id**. Records can span pages.
- **Documents**: JSON documents (meta, structs, functions, symbols) and
  **binary instance batches** (see below).

Because the file is just a sequence of documents, a database file is
effectively a **log**: loading replays documents in insertion order, and
incremental saves simply append more documents.

## Document kinds

| `kind` | Contents |
|---|---|
| `meta` | Format version and `next_type_id` — the loader takes the **latest** one |
| `struct` | A full struct definition: name, parent name, attributes (type, default), methods (real ASTs) |
| `function` | A standalone function (body stored as a real AST) |
| `instances` (binary) | A batch of up to 512 rows of one struct in the compact binary row format |
| `instance_removal` | Tombstone: `entity_id` — the loader drops the entity (and any pending row for it) when it reaches this document |
| `instance` (legacy v1) | JSON row: `struct_name`, `entity_id`, attribute map — still loaded for old files |
| `symbols` | A map of symbol name → entity id |

Two encodings deserve mention:

- **References** between instances are stored as `{"__ref__": entity_id,
  "__struct_type__": "..."}` markers in JSON documents, and as plain entity
  id varints in binary rows — so an object graph survives persistence
  without duplication.
- **ASTs** (method and function bodies, default expressions) are stored as
  structured nodes, *not* source text: a reloaded database re-executes
  original code without reparsing.

## Binary instance rows (v2)

Instance rows are encoded **positionally against the struct's declared
attribute types** (see `binary_cells.py`), so type information is stored
once in the schema instead of per value. Each row is:

1. a **presence bitmap** (one bit per column; missing attributes become
   absent bits instead of stored nulls),
2. an **escape bitmap** (columns whose payload does not fit the declared
   type fall back to JSON codec bytes for that row),
3. the packed values: little-endian ints of the declared width, 64-bit
   floats, length-prefixed UTF-8 strings, `DateTime` as 7×int32, `Color`
   as 4 bytes, struct references as entity id varints, arrays/dicts as
   varint counts plus recursively encoded elements, sets as varint
   counts plus elements in **sorted order**, queues as varint counts
   plus elements in serving order (priority queues normalized by
   priority, then element repr, for deterministic bytes), sparse
   spatial grids as varint counts plus (zigzag varint coordinates,
   element) pairs in **sorted coordinate order**, and tensors as a
   varint count plus the flat row-major elements (the shape lives in
   the schema type).

A typical typed row costs **~66 bytes** (versus ~1 KB in the legacy JSON
form, and ~76 bytes for an equivalent SQLite row).

## Lazy loading (v2)

`load_database` registers binary rows **without decoding them**. Each
struct table's instance mapping is a `LazyInstanceDict`: rows materialize
into `StructInstance` objects on first access and are cached. Consequences:

- **Open is O(meta)**: a large file opens in milliseconds and decodes
  nothing until a row is touched (point lookups cost ~0.1 ms).
- **Untouched rows re-dump byte for byte**: open → dump never decodes, so
  `compact` is near-free and `save_changes` only decodes rows it must
  diff.
- **References decode on demand**: traversing `jean.boss.name` decodes
  exactly the rows along the path (cycles are handled via stable shells).
- Full scans (`where`, `to_list`, aggregation) still decode every scanned
  row — laziness helps selective access, not full-table work.

### Memory model (buffer pool)

Rows are indexed, not copied: each batch document becomes a columnar
`BatchIndex` (parallel arrays of entity ids and payload spans, ~50
bytes per row measured), and payloads are reassembled one document at
a time through a bounded cache. Decoded instances live in a bounded
LRU (`RowDecoder.max_decoded`, default 100k):

- **Clean, unpinned instances evict freely** and re-decode from disk;
- **Dirty instances never evict** — they are the working set. Dirty is
  set by `set_attribute` and by in-place tensor mutations (caught via
  version-counter snapshots); cleared by the flush write-back.
- **`pin(elt)` keeps an instance resident** (e.g. the model during
  training); `unpin(elt)` releases it.
- Later appended batches override earlier rows for the same entity
  (replay-log semantics), so deltas stack safely.

## Full dump: `dump_database`

`dump_database(database, storage_path)` writes a **complete snapshot**:

1. One `meta` document (format version 2).
2. Struct definitions (in registration order, so type ids stay stable).
3. Function definitions.
4. Instance batches (binary, ≤512 rows per document per struct).
5. Symbol bindings.

The target file **must not exist** — snapshots are not appends. `Nadata.dump`
and `nadata dump` both use this path.

## Loading: `load_database`

`load_database(storage_path)` reconstructs the database:

1. Parses all documents (JSON and binary batches).
2. Registers struct shells, then parents/attributes/methods (so forward
   type references resolve).
3. Registers standalone functions.
4. **Registers binary rows lazily** (no decoding); legacy v1 JSON rows
   decode eagerly with reference placeholders.
5. Resolves cross-instance references (v1) or defers them to on-demand
   row decoding (v2).
6. Merges symbol bindings and settings in replay order.

Raises `ValueError` if the file has no recognizable meta document.

## Incremental saves: `append_database_changes`

`append_database_changes(database, storage_path)` diffs the live database
against the file and appends only what changed:

- New structs (or new attributes/methods on existing ones) as `struct`
  documents.
- New or changed instances as **binary batch documents**. Reference rows
  are decoded one at a time for the comparison, so a delta over a large
  lazy file only touches the changed rows.
- New or re-bound symbols as a `symbols` document.
- A fresh `meta` document (the loader takes the latest).

Returns the number of documents appended.

### The replay model

The loader never rewrites: it replays. A re-declared struct merges into
the earlier definition; a repeated entity id replaces the earlier
attributes; later `meta`/`symbols` documents override per key. This is
what makes appends safe — the file stays a valid log at all times.

## Compaction

`compact_database(storage_path, output_path=None)` loads the file and
dumps a clean snapshot (row payloads are copied without decoding when
untouched), then atomically replaces the target via `os.replace`. Returns
`(documents_before, documents_after)`; `count_documents(path)` counts
stored documents. `Nadata.compact()` and `nadata compact` wrap these.

## Database settings

The `meta` document carries a `settings` dict (persisted in every meta
document, merged in replay order). `loop_limit` is the first setting;
`Nadata.set_loop_limit(n)` and the `set_loop_limit(...)` builtin write it.

## Optimizer state sidecars

Model weights live in instance attributes and persist like any other
data. Optimizer machinery state (Adam moments, step counts, SGD
velocity, hyperparameters) is *user-side training junk* — it is
deliberately **not** stored in the database. Instead, optimizers
take a pickle sidecar file, written and loaded explicitly by the
user:

```text
opt = Adam(lr = 0.001);
...train...
opt.save_to_file("checkpoint.opt");      // moments + t + hyperparams
```

```text
opt = Adam(lr = 0.001);
opt.load_from_file("checkpoint.opt");    // resume exactly
```

The file is tagged and class-checked: loading Adam state into an SGD
is a `TypeError`. Only load files you trust (pickle). See
`examples/ex12_optimizer_checkpoint` for the resume-vs-cold-restart
demonstration.

## Python surface

```python
from nadata.core.database.persistence import (
    dump_database,
    load_database,
    append_database_changes,
    flush_dirty_instances,
    append_instance_removal,
    compact_database,
    count_documents,
)
```

## Memory API (scripts and Python)

| Call | Effect |
|---|---|
| `flush()` | Writes all dirty instances as delta rows; returns the count. Marks them clean. |
| `clear_cache()` | Evicts clean, unpinned instances from the decode caches; returns the count. |
| `pin(elt)` / `unpin(elt)` | Keeps / releases an instance in the decode cache. |
| `remove_from_db(elt)` | Removes the instance in memory and appends a tombstone (ids are never reused). |
| `db.flush()` / `db.clear_cache()` | Python equivalents on `NaDataDatabase`. |
| `db.remove_instance(instance)` | In-memory removal only (no tombstone). |

## CLI surface

```bash
nadata dump <project> <file.ndb>     # full snapshot
nadata info <file.ndb>               # structs, counts, settings
nadata compact <file.ndb> [output]   # rewrite without dead documents
```

## Schema evolution

A session folder re-merges its `.nastruct` files on every open. When
the new schema no longer matches persisted rows, reconciliation runs
after the merge and enforces two rules:

| Schema change | Effect on stored rows |
|---|---|
| Attribute removed | Value dropped — eagerly for decoded rows (the row is marked dirty, so the next flush / dump persists the removal), at decode time for pending rows (the stale column never materializes). |
| Attribute type changed | Mismatch **reported, not fatal**: the value is kept so the project opens and a migration script (`#manual`) can repair rows. The report lists struct, attribute, and entity id. |
| Attribute added | Nothing to do — the schema default materializes on first access. |
| Struct renamed | The old struct is untouched (still in the database); the new name creates a fresh table. |

The report surfaces through `FolderResult.reconciliation` and the
session summary, and stays live: lazy rows that decode later append
their findings to it.

```python
database, result = load_folder_session("./my_project")
print(result.reconciliation.summary())
# schema reconciliation:
#   dropped 2 stale attribute value(s) (Child: 2)
#   type mismatch: Child.age on entity 4194305: ...
```

Removed attributes vanish from the *file* the next time their row is
rewritten; untouched rows keep the old column on disk (harmlessly —
it is never decoded into the model) and disappear on the next
compaction that rewrites them.

## Invariants

- Entity ids and type ids are stable across dump/load round trips.
- `dump → load → dump` produces the same logical database; untouched
  lazy rows are copied verbatim.
- The loader accepts format version 1 (legacy JSON rows) and 2 (binary
  batches) in the same replay log.
- Struct definitions always precede their instance batches in the
  document order written by the dumper.
