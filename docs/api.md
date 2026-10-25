# API and CLI Reference

Nadata is usable as a Python library (`import nadata`) and as a shell tool
(`nadata`). This document covers both surfaces.

---

## Python API

Everything lives behind the `Nadata` facade:

```python
import nadata

app = nadata.Nadata.load("./examples/ex0_simple")
```

Values cross the boundary as **plain Python**: arguments are wrapped into the
runtime's typed `Value` objects automatically (lists and dicts infer their
element/key types from the first item), and results are unwrapped back to
plain Python (`int`, `float`, `str`, `bool`, `list`, `dict`).

### Loading

| Call | Behaviour |
|---|---|
| `Nadata.load(path)` | Auto-detects: a folder (project), a `.nastruct` file, or an `.ndb` database |
| `Nadata.open(path)` | Opens an existing `.ndb` file |
| `Nadata.load("./proj/")` | Loads `structs/*.nastruct` then `data/*.nadata`, executing scripts |

A lone `.nadata` file cannot be loaded — data requires its schema.

### Introspection

| Property / method | Returns |
|---|---|
| `struct_names` | `list[str]` — registered struct names |
| `function_names` | `list[str]` — registered standalone functions |
| `symbol_names` | `list[str]` — bound data symbols |
| `count(struct_name)` | `int` — instances of the struct **and its subclasses** |
| `total_instance_count()` | `int` — all instances |
| `summary()` | `str` — human-readable overview (used by `nadata info`) |
| `get(symbol)` | `StructInstance` bound to a data symbol |

### Reading and writing instances

```python
app.get_attribute("tom", "birthplace")        # -> 'Paris'
app.set_attribute("tom", "birthplace", "Lyon")
app.create_instance("Human", {"name": "Zoe"}, symbol="zoe")
```

`create_instance` applies schema defaults for omitted attributes; `symbol`
optionally binds the new instance into the symbol table (making it visible to
later code execution).

### Calling into the model

```python
app.call_method("tom", "older", app.get("jean"))   # tom.older(jean)
app.call_function("is_adult", app.get("tom"))      # is_adult(tom)
```

Positional and keyword arguments are both supported and type-checked against
the declaration.

### Queries

```python
app.count("Entity")   # polymorphic: counts subclasses too
app.query("Human")    # QuerySet — chainable
```

`QuerySet.where(...)` accepts a plain Python callable
(`lambda h: h.get_attribute("name").get() == "Tom"`) and supports chaining.

```python
qs = app.query("Person")
qs.order_by("age", descending=True).limit(5)   # sorted window
qs.where(lambda p: p.attributes["city"].get() == "Paris").exists()
qs.pluck("name")                     # raw payloads, in instance order
qs.sum_of("age")                     # min_of / max_of / average_of too
```

`order_by` validates the attribute against the schema and sorts stably by
the raw payload; aggregation helpers require numeric attributes. Attribute
names may be passed as plain strings (Python) or string values (language
side) — both are accepted.

### Geospatial layer

Coordinates, polygons, field layers, and routing are builtins (see
[syntax.md](syntax.md) §5.10); the Python side works on the same
tables:

```python
from nadata.core.types.home_types.coordinate import Coordinate

table = app.database.tables_by_struct_name["AdminRegion"]

# Session-scoped index over a geometry attribute; one index covers
# the whole struct hierarchy (subclass tables included)
index = table.create_spatial_index(
    attribute="boundary", lod=8, parent_attribute="parent",
    database=app.database,
)

found = index.locate(coord=Coordinate(43.6, 1.44, 6371008.8))
instance = index.get_instance(key=found)      # deepest containing row
hits = index.within(coord=coord, radius_m=8000.0)   # entity-id keys
ordered = sorted(ids, key=lambda i: index.sort_key(local_id=i, table=table))

# Routing networks: registered on both node and edge tables
place_table = app.database.tables_by_struct_name["Place"]
network = place_table.create_network(
    edges_table=app.database.tables_by_struct_name["Link"]
)
path = network.route(start=coord_a, end=coord_b, mode="car")
```

The language-side builtins (`spatial_index`, `locate`, `nearest`,
`within`, `network_index`, `route`) wrap the same machinery from
scripts; `get_by_nid(Base, "nid")` resolves subclass rows too.

### Executing raw source

```python
app.execute_schema("""
    class Weapon {
        name: Str = "Sword";
        @describe() -> Str { return self.name; }
    }
""")

app.execute_data('excalibur = Weapon(name = "Excalibur")')
app.execute('class Shield { defense: Int32 = 5; }')   # auto-detects schema vs data
```

- `execute_schema(text)` parses `.nastruct` source into the **live**
  database — new structs and functions are immediately available.
- `execute_data(text)` parses and executes `.nadata` source: instances are
  created and bound, scripts run immediately.
- `execute(text)` routes to one of the two above by detecting top-level
  `class` / `fn` declarations.

After any of these, new symbols and structs are visible to subsequent API
calls and queries.

### Persistence

```python
app.dump("store.ndb")          # full snapshot (file must not exist)
n = app.save_changes("store.ndb")   # incremental: append only the delta
app.set_loop_limit(100000)     # raise the per-loop fuel limit (persisted)
```

`save_changes` returns the number of delta documents appended (1 when nothing
but the meta header is written). `set_loop_limit` raises `LoopLimitError`
protection from 10,000 to the given per-loop iteration cap and persists the
value in the database settings. See [persistence.md](persistence.md).

---

## Sessions — folder projects

A project **folder** is a session: schema files (`.nastruct`, always
re-merged), plus `.nadata` scripts classified by their first-line directive
— `#init` (once per world), `#auto` (every open), `#manual` (explicit).
Opening a folder creates or reuses `<folder>/nadata.ndb` (the *world*) and
runs the pipeline. See syntax.md §7.7–7.8 for the language rules and
§7.9 for the I/O builtins.

```python
n = Nadata.open_folder("my_project")            # pipeline + session context
n = Nadata.open_folder("my_project", db_path="worlds/john.ndb")
                                                # --db: many worlds, one schema
n = Nadata.open_folder("my_project", run_auto=False)   # --no-auto (deterministic)
n = Nadata.open_folder("my_project", force_init=True)  # --force-init

n.execute_script_file("calibrate.nadata", {"target": "temp_living"})
                                                # run one #manual with params
```

Key guarantees:

- **init once**: `#init` scripts leave a content-hash marker in the world's
  settings; a failed init persists nothing (and a fresh world is not
  created). `force_init=True` re-runs them (nid upserts keep ids stable).
- **failed init aborts the open**: an existing world is reloaded from disk
  and `#auto` is skipped, so a broken seed can never reach the file.
- **stable identity**: `nid = "..."` declarations upsert in place — same
  entity id, passed attributes updated, the rest kept.
- **schema re-merge**: editing `.nastruct` files and reopening updates
  structs in place (new members added, re-declared members replaced).
- **explicit manual**: `#manual` scripts only run via
  `execute_script_file` / `:run` / `nadata run <folder> <script>`.

### Session CLI

| Command | Behaviour |
|---|---|
| `nadata open <folder>` | Run the pipeline, then drop into the console |
| `nadata run <folder> <script> [k=v ...]` | Pipeline + one `#manual` script with parameters |
| `nadata check <folder>` | Pipeline only, report, exit 0/1 (CI gate) |
| `--db <path>` | World database file (relative to the folder) |
| `--no-auto` / `--force-init` | Skip `#auto` / re-run `#init` |

### Session console commands

| Command | Behaviour |
|---|---|
| `:run file.nadata [k=v ...]` | Run a `#manual` script into the persistent scope |
| `:files` | List the session's scripts and directive signatures |
| `:reinit [file]` | Re-run `#init` scripts (optionally one file) |
| `:auto` | Re-run the `#auto` scripts now |
| `:flush` | Write all dirty instances to the world file |

Console state is **persistent across requests** (notebook semantics):
`:run` leaves its variables in the scope. Script **files** are stateless —
they run in fresh globals; the world database is the only channel between
scripts.

### I/O builtins

`read_text` / `write_text`, `read_lines` / `append_text`, `read_json` /
`write_json`, `read_csv` / `write_csv` (array of `dict[Str, Str]`,
stringly-typed by design — no inference), plus `exists`, `file_hash`,
`list_dir`, and `get_setting` / `set_setting`. Paths resolve against the
**project root** and are sandboxed to it: traversal above the root and
absolute paths raise `PermissionError` in session projects (without a
project root, the current working directory is used, unconstrained);
file effects are immediate (unlike the database, which is gated by
`flush()`).

```c
// Config-driven scripts
cfg = read_json("config.json");
kid.age = cfg["target_age"];

// Opt-in input freshness: re-import only when the input changed
if (get_setting("data_hash") != file_hash("input.csv")) {
    import_data();
    set_setting("data_hash", file_hash("input.csv"));
    flush();
}
```

`read_json` accepts objects of scalars, arrays of scalars, and scalars;
int/float mix as the number family (all-int stays Int64), and mixed
families or nesting raise with guidance. `set_setting` records a scalar
in the world's settings — persisted by the next `flush()`. See
[syntax.md](syntax.md) §7.9.

---

## Conversion tools

Two stdlib-only converters under `tools/` turn real-world data into
`.nadata` rows compatible with the ex13_earth schema (`Place` / `Link`
/ `AdminRegion`); both stream, so city-scale files work on a laptop.

**OpenStreetMap** (`tools/osm_to_nadata.py`) — OSM XML or `.osm.bz2`
extracts (Overpass API or Geofabrik):

```
python tools/osm_to_nadata.py map.osm map.nadata
```

- every node of a highway way becomes a shared `Place` junction and
  every consecutive node pair a `Link` segment, so the graph stays
  connected at real-world crossings (ways pass *through* junction
  nodes; endpoint-only edges would shred the network),
- `highway` class maps to transport modes (`motorway` car-only,
  `footway`/`steps`/... walk-only, the rest car + walk) and
  `maxspeed` becomes the car speed,
- named amenity / shop / tourism / leisure nodes (and building ways,
  centroid-positioned) become typed `Place` rows; closed
  `boundary=administrative` ways become `AdminRegion` rows,
- ways referencing nodes outside the extract are dropped and counted.

**GeoJSON** (`tools/geojson_to_nadata.py`) — FeatureCollections to
`AdminRegion` rows (tested against the real france-geojson
departements file):

```
python tools/geojson_to_nadata.py areas.geojson areas.nadata     --level Departement --parent-nid france
```

`--parent-nid` binds every row to a parent via `get_by_nid`; nid
lookup walks the inheritance hierarchy, so parents declared in other
leaf tables (Country, Region) resolve.

See `tests/test_osm.py` for the committed real-data fixtures
(central Toulouse OSM extract, the 96 metropolitan French
departements) and the full pipeline assertions: convert, load,
route car/walk, locate, radius queries.

---

## CLI

```
nadata <command> ...
```

| Command | Arguments | Behaviour |
|---|---|---|
| `nadata run <path>` | project folder, `.nastruct`, or `.ndb` | Loads everything and executes the data scripts |
| `nadata info <path>` | same | Prints the database summary |
| `nadata dump <source> <dest>` | source project, destination `.ndb` | Loads and persists a full snapshot |
| `nadata compact <path> [output]` | an `.ndb` file, optional output path | Rewrites the file to its live state (defaults to in place) |
| `nadata console <path>` | same | Opens an interactive console over the loaded database (see below) |

Exit code is `0` on success, `1` on error (the error message goes to
stderr). The console script is installed as `nadata` via `pip install -e .`;
the same commands work from the repository root as
`python nadata.py <command> ...` without installation.

### Examples

```bash
nadata run examples/ex0_simple
nadata info examples/ex3_collections
nadata dump examples/ex0_simple store.ndb
nadata info store.ndb
```

---

## Interactive console

`nadata console <path>` loads a project or database and opens a
read-eval-print loop over it — works identically over a project folder and
a persisted `.ndb` file:

```text
$ nadata console examples/ex0_simple
Nadata console - 6 struct(s), 6 instance(s), 2 function(s).
nadata> tom.name
      ... 
Tom
nadata> adults = Human.where((h: Human) -> h.birthplace == "Paris");
      ... 
nadata> for(h: Human in adults){
      ...   print(h.name);
      ... }
      ... 
Tom
Jean
Kate
nadata> :quit
```

- Requests end with a **blank line** (double ENTER): source lines accumulate
  in a buffer, and the blank line runs the whole request — this is what
  makes multi-line loop and if bodies possible. `:commands` and `:quit`
  stay immediate on a single ENTER; Ctrl-C cancels the current request.
- A request holding a **bare expression** echoes its formatted value (type
  handles render as `<Type ...>`, instances as `<Struct id=N>`).
- **Declarations and statements** execute against the live database — created
  instances become queryable symbols.
- **All state persists across requests**: script variables, new instances,
  and schema changes.
- `print(...)` and other None-returning calls (e.g. `setattr`) stay silent.
- Meta commands: `:help`, `:info`, `:classes`, `:symbols`, `:quit` (also
  `:q`, `exit`, `quit`). Ctrl-D exits gracefully.
- Errors print as `error: ...` and keep the session alive.

The same loop is available from Python:

```python
import nadata
from nadata.console import NadataConsole

app = nadata.Nadata.load("store.ndb")
NadataConsole(app=app).run()
```

Or drive it programmatically line by line with `console.execute_line(line)`,
which returns the echoed value for expression lines and `None` otherwise.

---

## Error behaviour

| Situation | Raised |
|---|---|
| Unknown struct in `create_instance` | `KeyError` |
| Unknown symbol in `get` / `get_attribute` | `KeyError` |
| Unresolvable identifier in executed code | `NameError` |
| Loading a non-Nadata `.ndb` file | `ValueError` |
| `save_changes` on a missing file | `ValueError` (use `dump` first) |
| Method call with undeclared argument | `TypeError` |
| Parameter or local shadowing an attribute | `SyntaxError` at resolve time |
