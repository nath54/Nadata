# Memory & Streaming Plan

Large-dataset management for Nadata: bounded memory during training,
streaming entity datasets from disk, and explicit cache control.

Planning document — no implementation contract beyond the sequencing at
the end. Companion to the autograd plan; orthogonal to it (tape work
does not touch the storage layer).

---

## 1. Current state (verified against the code)

### 1.1 Open path materializes the whole file

`load_database` calls `record_store.read_all_documents()`, which scans
every page and reassembles **the entire file** into one
`dict[int, bytes]`. Every instance row then registers as a `PendingRow`
holding a `payload: bytes` slice. Consequences:

- The whole dataset is resident as raw bytes from the moment the file
  opens, decoded or not.
- The source document dict is discarded after load, so **pending rows
  are no longer re-readable from disk**. Dropping a payload loses data.

### 1.2 The decode cache is unbounded

`RowDecoder.decoded` caches every materialized `StructInstance`
forever. No eviction, no dirty tracking, no way to distinguish mutated
instances from clean ones. Decoded Python objects are typically much
larger than their encoded rows.

### 1.3 The PageManager LRU is defeated in practice

`PageManager` bounds its page cache (256 x 64KB) correctly, but
`read_all_documents` reassembles everything out of it into an
unbounded dict during the scan. The bounded component never survives
the load.

### 1.4 Delta saves load the file again

`append_database_changes` loads a **full reference copy** of the
target file to diff against. Saving a checkpoint is O(file) in memory
and time — on a 10 GB dataset, appending a few changed rows holds
roughly twice the file.

### 1.5 What already works for us

- Chunked records (`RecordStore`), lazy row registration, and
  on-demand decode with cross-reference resolution (the "shells"
  mechanism) all exist.
- Mutation hooks already run on every in-place write
  (`set_attribute`, `_assign_tensor*`, tensor `set_at` / `fill`) —
  they already bump `version` counters for autograd.

---

## 2. Principles

1. **Lifetime is a status, not an id format.** Temporary or cached
   instances use normal `EntityId`s from the normal counter; ids are
   never reused, so a stale reference resolves to `None` instead of
   impersonating a new entity.
2. **Eviction is persistence.** Memory is freed only by flushing dirty
   state to the delta log and dropping clean state that can be
   re-read. There is no other correct way to free memory here.
3. **Read-only data streams; written data stays.** Dataset rows are
   read-only during training, so they stay clean and evict freely.
   The model and optimizer are mutated every step, so they stay
   resident by their own dirtiness — no reload cost, no I/O churn.
4. **The hot path must not hit disk.** The model instance is touched
   every batch. It must be pinned explicitly (see 4.4) so eviction
   never even considers it.
5. **No new dependencies.** The buffer pool is built from the existing
   page/record machinery, pure stdlib.

---

## 3. Workstream 1 — Index, don't reassemble

Replace the load-time full reassembly with a document index:

- `RecordStore.document_locations()` returns
  `dict[doc_id, list[(page_id, slot)]]` (chunk locations).
- `RecordStore.read_document(doc_id)` reassembles one document on
  demand through the `PageManager` (which already caches pages under
  its LRU bound).
- The loader stores locations in `PendingRow` instead of payload
  bytes. Decoding a row reassembles its own document at that moment.

Effects:

- Open becomes O(index): one small tuple per row, no payload bytes.
- Any row is re-readable at any time — the precondition for eviction
  and for `clear_cache()`.
- The transient peak during load drops from O(file) to O(page cache).

---

## 4. Workstream 2 — Dirty tracking and pinned instances

### 4.1 Dirty flag

`StructInstance` gains a `dirty: bool` flag (default `False`), set by
every mutation hook: `set_attribute`, `_assign_tensor`,
`_assign_tensor_flat`, tensor `set_at` / `fill`, and queue/dict/set
in-place helpers. The same sites already bump tensor version counters.

No write-through: mutation is still memory-only until an explicit
save. Training-loop throughput is untouched.

### 4.2 Pinned flag

`StructInstance` gains a `pinned: bool` flag (default `False`). Pinned
instances are never evicted by the cache and never considered by
`clear_cache()`. Use it for anything touched every batch:

```text
model = Model();
pin(model);                        // stay resident, zero decode cost
opt = SGD(lr = 0.05);
```

`pin` / `unpin` are builtins (thin wrappers over the instance flags,
like `set_learnable`). The pinned flag is a runtime status — it is
not serialized.

### 4.3 Why dirty data needs no explicit pin

A dirty instance is by definition part of the working set; evicting it
would cost a delta append plus a re-decode on next touch. The default
eviction policy (4.1 of Workstream 3) simply **does not evict dirty
instances** — they accumulate until `flush()` or a checkpoint. Clean
read-only data (the dataset) evicts freely. The model ends up resident
for free, but `pin(model)` documents intent and protects the
clean-but-hot case (e.g. between epochs when nothing was written).

---

## 5. Workstream 3 — Bounded decode cache, eviction, clear_cache()

### 5.1 The cache

`RowDecoder.decoded` becomes an LRU with a configurable bound
(`max_decoded_instances`, generous default, e.g. 100k instances or a
byte-based estimate). Pinned instances bypass the LRU ordering.

### 5.2 Eviction policy

When the bound is exceeded, evict least-recently-used **clean**
instances:

- Clean: drop the decoded object; the row stays re-readable through
  its index entry (Workstream 1).
- Dirty: do not evict — the working set is allowed to exceed the
  bound rather than silently appending I/O. (A future refinement can
  add a hard-limit mode with write-back; the default favors
  predictability.)

### 5.3 clear_cache() and flush()

- `clear_cache()` (builtin + `app.clear_cache()`): drop all clean
  decoded instances (pinned excluded). Pinned and dirty instances
  stay. Memory returns to index + working set.
- `flush()` (builtin + `app.flush()`): append every dirty instance to
  the delta log (Workstream 4), then mark them clean. After a flush,
  `clear_cache()` can free everything unpinned.
- `checkpoint(path)` convenience (optional): flush + full compacted
  dump of the current state.

---

## 6. Workstream 4 — Dirty-only delta appends and tombstones

- With dirty tracking, `append_database_changes` no longer loads the
  reference file for **instances**: it appends exactly the dirty rows
  in entity-id order. Struct and symbol deltas keep the existing diff
  path (rare and small).
- **Tombstones**: removal (the `remove_from_db_from_id` design from
  the lifecycle discussion) writes a delete record to the same delta
  log so compaction does not resurrect removed instances. The loader
  applies deletes in insertion order like every other record.
- `compact_database` rewrites the file without deltas/tombstones.
  Schedule it after long training runs.

Together: checkpointing a training run becomes O(dirty), not
O(file).

---

## 7. Workstream 5 — Streaming entity datasets

### 7.1 The pattern

Dataset rows are entities of a dedicated struct, stored in the `.ndb`;
training reads them through the decode cache:

```text
class DatasetRow {
    x: Tensor[Float32, 2] = zeros();
    y: Tensor[Float32, 1] = zeros();
}
```

`get_batch_of_partition` permutes and slices its input list — if the
input is an **array of entity references**, the returned batch is
`Array[DatasetRow]`, and each field access (`r.x`, `r.y`) decodes
that row on demand through the bounded cache. Memory during training
is O(batch + model + pinned).

```text
rows = DatasetRow.all();          // entity references, not payloads
total = get_partition_iteration_count(rows, 42, [0.8, 0.2], 32, 0);
for (it: Int64 = 0 :: it < total :: it++) {
    batch = get_batch_of_partition(rows, 42, [0.8, 0.2], 32, 0, it);
    for (r: DatasetRow in batch) {
        x: Tensor[Float32, 2] = r.x;   // decoded here, cached, evictable
        ...
    }
}
```

The in-memory `Array[Array[Float32]]` path stays unchanged for small
data; both flows share the same partition helper.

### 7.2 Verification needed first

- Entity-reference arrays flowing through loader scopes, the
  partition helpers' type inference, and `for (... in ...)` loops.
- That decode-on-access through `r.x` resolves cross-references the
  same way as direct `get_by_id` access.
- QuerySet → array conversion for `.all()` results passed to the
  partition helpers.

If any of these need glue, it is loader-level, not storage-level.

### 7.3 Acceptance scenario

A `.ndb` with ~1M `DatasetRow` instances trains the Phase B MLP with
peak memory bounded by (model + pinned + batch + page cache) —
verified by asserting the decode-cache size stays under its bound
during the run.

---

## 8. Sequencing

| Order | Item | Size | Depends on |
|---|---|---|---|
| 1 | RecordStore index: `document_locations` / `read_document`, location-based `PendingRow` | medium | — |
| 2 | Dirty flag on mutation hooks (+ tests) | small | — |
| 3 | LRU decode cache, clean-only eviction, `pin`/`unpin` | moderate | 1 |
| 4 | `clear_cache()` / `flush()` builtins + Python API | small | 2, 3 |
| 5 | Dirty-only delta appends + tombstone records | small | 2 |
| 6 | Entity-row streaming: verify the 7.2 list, docs, example | small (verify first) | 1, 3 |

Suggested pairing with the roadmap:

- **Phase B (autograd)** proceeds independently; step 6's
  verification can slot into its acceptance tests since Phase B wants
  a trained-MLP scenario anyway.
- **Lifecycle removal** (`remove_from_db_from_id`) lands together
  with step 5 — they share the record-log changes.

## 9. Open points

1. Byte-based vs instance-count cache bound (start with count; add a
   byte estimate once real dataset profiles exist).
2. Whether `flush()` should also compact when deltas exceed a ratio —
   probably a policy knob on `compact`, not automatic.
3. A chunked big-tensor column store (one dataset struct holding a
   huge tensor, sliced through persistence) is a possible later
   optimization for fixed-shape rows; entities first, profile before
   building it.
