# Session & Identity Plan

The terminal workflow: open a project folder, get a seeded database, run
parameterized scripts, inspect in the console — without ever duplicating
data or re-running expensive work. Two phases: **identity** (`nid`) and
**sessions** (folder loading with `#init` / `#auto` / `#manual`).

> **STATUS: IMPLEMENTED** — Phase 1 (`2ff58a8`): nid identity, upsert,
> lookups, persistence, tombstone bookkeeping, `str()` builtin. Phase 2:
> session pipeline (folder loader, `#init`/`#auto`/`#manual`, markers,
> failed-init rollback), schema re-merge in the parser, parameter preludes,
> CLI verbs (`open`/`check`/`run` + `--db`/`--no-auto`/`--force-init`) and
> console commands (`:run`/`:files`/`:reinit`/`:auto`/`:flush`).
> **Not implemented:** I/O builtins (§3.5, next arc).
>
> Deviations from the text below: a failed `#init` on an existing world
> reloads the world from disk and skips `#auto` (stricter than "no flush"
> — pass-1 shells from the broken script can never reach the file), and
> `LazyInstanceDict.__len__` counts decoded ∪ pending (row-backed decoded
> entities stay listed as pending).

---

## 1. Verified current state

- `NaDataLoader.load_folder` already loads all `.nadata` files in a folder
  with a shared symbol scope; `Nadata.load` dispatches folder / `.nastruct`
  / `.ndb` paths. There is no once-semantics: re-loading re-creates.
- `#` is already a comment character (CommentsRemover strips `#` lines), so
  a directive first line is safely ignored by today's parser — extraction
  happens on the raw text before parsing.
- Instance creation has exactly two paths: declaration statements
  (pass 1 creates empty + symbol, pass 2 binds attributes) and runtime
  constructor calls through `StructRuntimeBinding.__call__`. Both funnel
  through `database.create_instance` + `set_creation_attributes`.
- `database.symbols: dict[str, EntityId]` persists via the symbols doc
  (`{"kind": "symbols", "symbols": {...}}`); the loader merges. Entity ids
  are never reused; `remove_instance` + tombstones exist (memory plan).
- The buffer pool (memory plan) gives: dirty-flag transactions (unflushed
  rows vanish), `flush()` as the persistence gate, O(index) open.

## 2. Phase 1 — identity: `nid`

### 2.1 Semantics

- `Child(nid = "enfant1", ...)` binds the instance to a **named id** in a
  **per-struct namespace**. Without `nid`, instances are anonymous
  (loops create freely).
- Re-declaration with an existing nid is a **partial update in place**:
  passed attributes are re-bound (consts included — upsert is
  re-construction), absent attributes keep their value, and the
  **entity id is stable**. The dirty flag routes the change through
  `flush()`; nothing is duplicated.
- `nid` is immutable after creation. `remove_from_db` drops the nid
  entry; a later re-declaration creates a fresh instance with a **new**
  id (ids are never reused).
- `nid` values are dynamic expressions (`nid = "child_" + str(i)`), so
  loops can name instances from data.

### 2.2 Storage & lookup

- `database.nids: dict[str, EntityId]` keyed `"{Struct}/{nid}"`, plus a
  reverse map for O(1) `elt.nid` and tombstone drops. Persisted in the
  symbols document as `"nids"` (absent in old files → empty; loader
  merges, so full-map rewrites are safe).
- `StructInstance.nid: Optional[str]` — engine metadata on the instance,
  restored at decode via a provider callback (never stored in rows).
- Lookups: `get_by_nid(Struct, name)` → instance or **None** (missing is
  not an error); `first(Struct)` / `last(Struct)` sugar over the base
  QuerySet; `nids(Struct)` → sorted nid strings; `elt.nid` → the nid or
  None payload. Anonymous instances are addressed by `elt.id` + queries —
  nid machinery is strictly opt-in and inert when unused.

### 2.3 Acceptance

- Upsert keeps entity id stable across re-runs (declaration AND loop
  paths); partial update keeps unpassed attributes.
- nid map survives dump/load and incremental flush; decoded instances
  expose `.nid`.
- Tombstone removes the nid entry; post-tombstone re-declaration yields
  a new id.
- `get_by_nid` missing → None; `first`/`last`/`nids` match QuerySet.

## 3. Phase 2 — sessions: folders, directives, parameters

### 3.1 Folder model

- **Folder = the system; `nadata.ndb` = one world of its data.**
  `.nastruct` files (recursive, sorted) are re-parsed and merged on every
  open — declarative, idempotent. The `.ndb` holds instances.
- `--db path` points the world file elsewhere (relative to the folder);
  same system, many independent universes. No folder given → CWD is the
  project root (loud banner when creating `nadata.ndb` there).
- Directives are the **first line** of a `.nadata` file: `#init`,
  `#auto`, `#manual(...)` (with a parameter signature). Unmarked files
  are `#manual` with an empty signature — **automatic execution is
  opt-in**.

### 3.2 The pipeline (`nadata open <folder>`)

1. Scan and classify files; open the `.ndb` (lazy, O(index)) or create it.
2. Merge all schemas.
3. `#init` files (sorted): run once per world — marker
   `settings["init_done"]["<relpath>:<sha256>"]`. On success: write the
   marker and flush (fresh worlds are dumped whole). On failure: report,
   no marker, unflushed rows vanish (free transaction). `--force-init`
   re-runs.
4. `#auto` files (sorted): run every open. Nothing persists unless the
   script calls `flush()` — the write gate. Failures are reported, the
   console still opens. `--no-auto` skips them (tests/CI).
5. `#manual` files: never run automatically. Invoked with
   `:run script.nadata k=v ...` (console) or
   `nadata run <folder> <script> k=v ...` (headless).
6. Console attaches (`nadata open`), or exits (`nadata check`).

### 3.3 Parameterized `#manual`

- Signature: `#manual(lr: Float = 0.05, epochs: Int = 40, tag: Str)`.
  Scalars only (`Int`, `Float`, `Str`, `Bool`); defaults are literals;
  params without defaults are required (error lists the signature).
- Params are pre-bound typed variables in the script scope (loader
  prelude). Invocation coerces or errors — CLI strings coerce per the
  declared type; malformed values are hard errors before any statement
  runs. `#init`/`#auto` files reject parameter declarations at load.
- The directive line doubles as documentation: `:files` prints each
  manual script with its signature.

### 3.4 Console & CLI surface

```
nadata open <folder> [--db DB] [--no-auto] [--force-init]   # pipeline → REPL
nadata run <folder> <script.nadata> [k=v ...] [--db DB]     # headless #manual
nadata check <folder> [--db DB] [--no-auto]                 # CI gate (exit 1 on error)

:run train.nadata lr=0.1    :files            :reinit [file]
:flush                      :auto             (:info/:classes/:symbols exist)
```

`Nadata.load(folder)` becomes pipeline-aware: any directive marker or an
existing `nadata.ndb` switches the folder to session semantics (legacy
behavior preserved for directive-free folders, so golden tests hold).

### 3.5 I/O builtins (with the session layer)

| Builtin | Shape |
|---|---|
| `read_text` / `write_text` | whole-file Str; overwrite on write |
| `read_lines` / `append_text` | array of Str / append one line |
| `read_json` / `write_json` | dicts/arrays of scalars (overwrite on write) |
| `read_csv` / `write_csv` | array of `dict[Str, Str]`; stringly-typed by design |
| `exists` / `file_hash` / `list_dir` | Bool / Str hash / array of Str |
| `get_setting` / `set_setting` | opt-in freshness markers (`file_hash`) |

Paths resolve against the project root (one base, no ambiguity). File
effects are immediate — DB writes stay gated by `flush()`; scripts that
need all-or-nothing write files last.

### 3.6 Acceptance

- First open creates `nadata.ndb`, runs init, marker written; second open
  skips init, runs auto; `#manual` never runs on its own.
- Failed init leaves no marker and no rows (memory-model transaction).
- `:run`/headless coerce params; required-missing → signature error.
- `--no-auto` makes opens deterministic for tests; `check` exits 1 on
  auto/init errors.
- I/O builtins round-trip text/JSON/CSV against the project root; the
  quiz scenario works: one folder, `--db` per world, independent markers.

## 4. Sequencing

| Step | Item | Depends on |
|---|---|---|
| 1 | `nid` identity layer + lookups + tombstone bookkeeping | — |
| 2 | Folder pipeline, directives, console/CLI verbs | 1 (upsert safety) |
| 3 | `#manual(...)` parameters | 2 |
| 4 | I/O + settings builtins | 2 |
