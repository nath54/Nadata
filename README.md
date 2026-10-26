# Nadata

[![CI](https://github.com/nath54/Nadata/actions/workflows/ci.yml/badge.svg)](https://github.com/nath54/Nadata/actions/workflows/ci.yml)

**Nadata** is a declarative data schema and object-graph framework with its own
language. You describe your data model in `.nastruct` schema files, populate it
in `.nadata` data files, and get a typed, queryable, persistent object graph —
usable from the command line or as an embedded Python library.

The same language models geospatial domains end to end: coordinates on real
or fictional planets, hierarchical regions, road networks with transport
modes, buildings with floor-by-floor interiors, and planet-scale spatial
indexing — one `route()` can walk from a street to an upstairs bedroom.

```text
# structs.nastruct — the model
class Human {
    name: Str;
    birthday: DateTime = DateTime();

    @age(now: DateTime = DateTime.now()) -> DateTime {
        return now - self.birthday;
    }
}

# data.nadata — the data
tom = Human(name="Tom", birthday=DateTime(d=12, m=4, y=2003));

print("Tom is an adult:", tom.age(DateTime.now()).year >= 18);
Array[Human] adults = Human.where((h: Human) -> h.age(DateTime.now()).year >= 18);
```

## Features

- **Typed schema language** — structs, inheritance, attributes with defaults,
  `const` bind-once attributes, methods and computed properties, standalone
  functions; `import "...";` shares schema libraries across projects
- **String library and hash indexes** — typed string members (`split`,
  `join`, `replace`, `contains`, ...), and opt-in per-attribute hash
  indexes (`index(Human, "email")` + `get_by(Human, "email", x)`) with
  automatic insert / update / removal maintenance — O(1) equality
  lookups instead of full scans
- **Rich type system** — sized integers/floats (`Int32`, `UFloat16`, ...),
  `Str`, `Bool`, `DateTime`, `Color`, `Array[T]`, `Dict[K, V]`, `Set[T]`,
  `Queue[T]` / `PriorityQueue[T, P]`, sparse spatial grids
  (`GridSquare2D[T]`, `GridHex2D[T]`, `GridCube3D[T]`), shape-checked
  `Tensor[T, d1, ...]` with symbolic dims bound to `const` attributes,
  entity references; indexing (`arr[0]`,
  `dict["k"]`) and string concatenation
- **Error handling** — `raise "message";` and `try { ... } catch (err) { ... }`
  in scripts and method bodies; runtime errors (indexing, I/O builtins)
  are catchable too, while `return` / `break` / `continue` pass through
- **Declarative data files** — forward and circular references, script
  statements, `print`, chained polymorphic queries
- **Runtime introspection** — `elt.id`, `type(x)`, `elt.list_attr()` /
  `list_methods()`, schema-constrained `getattr`/`setattr`, and database-wide
  `schema_classes()` / `schema_functions()`
- **Persistence** — whole-database dumps, dirty-only delta appends, and
  removal tombstones to a single `.ndb` file
- **Bounded memory** — disk-backed lazy rows (~50 bytes/row index), a
  bounded LRU decode cache with clean-only eviction, `pin`/`unpin`,
  `flush`/`clear_cache`, and `remove_from_db` from scripts — stream
  millions of rows while the model stays hot
- **Autograd and training** — tape-based reverse-mode differentiation on Float
  tensors (arithmetic, `matmul`, reductions, shape ops, `gather`, slices,
  `relu`), seeded data partitioning for train/valid/test splits, and built-in
  `SGD` — train models end-to-end in `.nadata` scripts
- **Geospatial layer** — `Coordinate` (geodetic lat/lon + radius from the
  planet's center, spheres and spheroids), `Polyline` / `Polygon` geometry
  with geodesy builtins (`distance`, `bearing`, `area`, `interpolate`, ...),
  `spatial_index` with hierarchical `locate` / `nearest` / `within` queries
  (one index serves a whole struct hierarchy via entity ids),
  `GeoGrid[T]` field layers with bilinear `sample` and sea-level
  registration (`altitude` reads the interpolated sea surface), and
  `network_index` / `route(from, to, mode)` with distance or time costs,
  mode filtering, and doors-and-stairs-as-edges indoor routing
- **Sessions & identity** — a project folder is a session: `#init` seeds once
  per world, `#auto` syncs every open, `#manual` scripts take parameters
  (`#manual(lr: Float = 0.05)`), and `nid = "name"` gives instances stable,
  updatable identity (`get_by_nid`, `nids`, `first`/`last`) — schema files
  re-merge on every open, and schema evolution is reconciled against
  persisted rows (removed attributes are dropped; type changes are
  reported, never fatal). I/O builtins (`read_json`, `write_csv`,
  `file_hash`, ...) resolve against the project root, with
  `get_setting`/`set_setting` freshness markers
- **Interactive console** — `nadata console <path>` REPL over any loaded
  database, with persistent state and expression echoing
- **Python API** — load, query, mutate, execute raw source, persist
- **Zero dependencies** — pure Python 3.10+

## Installation

```bash
pip install nadata          # from PyPI
```

Or from source:

```bash
git clone <repository-url>
cd Nadata
pip install -e .
pip install -e .[dev]   # optional: pytest + mypy for development
```

### Native bindings (C/C++, Java, Kotlin)

For embedding from other languages there are two paths — both fully
self-contained (they carry the engine + embedded CPython):

```sh
# C/C++ (and any language): install the staged bundle next to your project
curl -fsSL https://raw.githubusercontent.com/nath54/Nadata/main/scripts/install.sh | sh
# Windows:
irm https://raw.githubusercontent.com/nath54/Nadata/main/scripts/install.ps1 | iex
```

```kotlin
// JVM (Java/Kotlin): one fat jar, self-extracting on first run
NadataBundled.install()
```

See [docs/distribution.md](docs/distribution.md) (setup) and
[docs/llm/INDEX.md](docs/llm/INDEX.md) (self-contained API knowledge
pack for JVM app development). The C++ header-only wrapper ships in
the bundle as `nadata.hpp`; the ABI is documented in
[docs/native.md](docs/native.md).

## Quick start

```bash
# Run a project (schema + data) and execute its scripts
nadata run examples/ex0_simple

# Print a summary of a project or a saved .ndb database
nadata info examples/ex0_simple
nadata info store.ndb

# Persist a project to a single database file
nadata dump examples/ex0_simple store.ndb

# Explore it interactively (works on folders and .ndb files)
nadata console examples/ex0_simple
nadata console store.ndb
```

Or from Python:

```python
import nadata

app = nadata.Nadata.load("./examples/ex0_simple")

print(app.get_attribute("tom", "birthplace"))   # Paris
print(app.count("Human"))                        # 3

app.dump("store.ndb")                            # persist
```

### Planet-scale data

The geospatial layer in action (from `examples/ex13_earth`):

```text
# structs.nastruct
class AdminRegion { name: Str; parent: AdminRegion; boundary: Polygon; }
class Place { name: Str; kind: Str; position: Coordinate; }
class Link {
    name: Str; from_place: Place; to_place: Place;
    modes: Array[Str] = []; speed: Dict[Str, Float64] = {};
    shape: Polyline;
}

# data.nadata
spatial_index(AdminRegion, "boundary", lod = 8, parent = "parent");
network_index(Place, Link);

here = locate(AdminRegion, Coordinate(43.6047, 1.4442, 6371008.8));
print("located in:", here.name);            # deepest containing region

path = route(Coordinate(43.6293, 1.3638, 6371008.8),   # Blagnac airport
             Coordinate(43.6047, 1.4442, 6371008.8),   # the Capitole
             "car", cost = "time");
print("drive time s:", path.cost);
```

Coordinates carry the **radius from the planet's center** (not an
altitude), resolved from per-world planet settings — so the same model
describes Earth, Mars, or a fictional world.

## Documentation

| Document | Contents |
|---|---|
| [docs/tutorial.md](docs/tutorial.md) | Step-by-step tutorial: your first project |
| [docs/syntax.md](docs/syntax.md) | Full language reference: rules and syntax |
| [docs/architecture.md](docs/architecture.md) | How the engine works, stage by stage |
| [docs/api.md](docs/api.md) | Python API and CLI reference |
| [docs/native.md](docs/native.md) | Native ABI: embedding the engine from C/C++/any host language |
| [docs/distribution.md](docs/distribution.md) | One-line setup: install scripts, fat jar, CMake package |
| [docs/llm/INDEX.md](docs/llm/INDEX.md) | Self-contained knowledge pack for building apps on the JVM bindings |
| [docs/persistence.md](docs/persistence.md) | The `.ndb` format, dumps and incremental saves |
| [docs/benchmarks.md](docs/benchmarks.md) | Engine stage profiling, synthetic benchmarks, SQLite comparison |
| [docs/development.md](docs/development.md) | Code style, testing, contributing |

## Examples

The `examples/` folder contains self-contained projects, from basics to
edge cases:

| Example | Demonstrates |
|---|---|
| `ex0_simple` | Family graph: inheritance, methods, forward refs, queries |
| `ex1_primitives` | Every sized integer/float type and boundary values |
| `ex2_inheritance` | Circular references, polymorphic queries |
| `ex3_collections` | Nested arrays, dictionaries, `Color`, `DateTime` math |
| `ex4_expressions` | `if`/`else`, boolean operators, local declarations |
| `ex5_edge_cases` | Parser edge cases: comments, spacing, empty constructs |
| `ex6_loops_collections` | Loops (`for-in`, C-style, `while`), fuel guards, collections |
| `ex7_queues_sets` | FIFO / priority queues, typed sets, membership probes |
| `ex8_grids_tensors` | Sparse spatial grids (square/hex/cube), shape-checked tensors |
| `ex9_constants` | `const` bind-once attributes, symbolic tensor dims |
| `ex10_cnn` | Conv1d/conv2d, pooling, and a basic convolutional neural network |
| `ex11_transformer` | A minimal transformer block: attention, LayerNorm, feed-forward, `topk` sampling |
| `ex12_optimizer_checkpoint` | Training a gelu network with Adam, checkpointing optimizer state to a sidecar file, and resuming from it |
| `ex13_earth` | Planet Earth: the full geospatial layer on real data — admin regions, road + indoor routing, elevation grids; `#init` / `#auto` / `#manual` session directives and parameterized scenarios |

Real-world data converters ship under `tools/` (stdlib only, streaming):
`tools/osm_to_nadata.py` turns OpenStreetMap extracts into routable
networks, and `tools/geojson_to_nadata.py` converts GeoJSON polygons into
admin-region rows. Both are exercised against committed real fixtures
(central Toulouse, the French départements) in `tests/test_osm.py` —
measured numbers for a 10 km × 8 km slice of Toulouse are in
[docs/benchmarks.md](docs/benchmarks.md).

## Editor support

Syntax highlighting for `.nastruct` and `.nadata` files ships as a
VS Code / VSCodium extension in
[`editors/vscode-nadata`](editors/vscode-nadata). `.vsix` packages are
not stored in git — build one and install it:

```bash
cd editors/vscode-nadata
npx -y @vscode/vsce package --allow-missing-repository   # -> nadata-support-<version>.vsix
code --install-extension nadata-support-<version>.vsix
```

Packaged releases attach the current `.vsix` to the matching git tag.

A sync test (`tests/test_grammar.py`) keeps the grammar aligned with the
language: it fails whenever a builtin type, keyword, or scope is missing.

## Development

```bash
python -m pytest tests/ -q          # run the test suite
python -m mypy nadata/              # type check (strict)
python -m scripts.inspect_parse examples/ex0_simple   # inspect the parse IR
```

## License

MIT — see [LICENSE](LICENSE).
