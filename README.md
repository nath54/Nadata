# Nadata

[![CI](https://github.com/nath54/Nadata/actions/workflows/ci.yml/badge.svg)](https://github.com/nath54/Nadata/actions/workflows/ci.yml)

**Nadata** is a declarative data schema and object-graph framework with its own
language. You describe your data model in `.nastruct` schema files, populate it
in `.nadata` data files, and get a typed, queryable, persistent object graph —
usable from the command line or as an embedded Python library.

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
  functions
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
git clone <repository-url>
cd Nadata
pip install -e .
pip install -e .[dev]   # optional: pytest + mypy for development
```

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

## Documentation

| Document | Contents |
|---|---|
| [docs/tutorial.md](docs/tutorial.md) | Step-by-step tutorial: your first project |
| [docs/syntax.md](docs/syntax.md) | Full language reference: rules and syntax |
| [docs/architecture.md](docs/architecture.md) | How the engine works, stage by stage |
| [docs/api.md](docs/api.md) | Python API and CLI reference |
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

## Editor support

Syntax highlighting for `.nastruct` and `.nadata` files ships as a
VS Code / VSCodium extension in
[`editors/vscode-nadata`](editors/vscode-nadata):

```bash
code --install-extension editors/vscode-nadata
```

A sync test (`tests/test_grammar.py`) keeps the grammar aligned with the
language: it fails whenever a builtin type, keyword, or scope is missing.

## Development

```bash
python -m pytest tests/ -q        # run the test suite
python -m mypy src/               # type check (strict)
python -m scripts.inspect_parse examples/ex0_simple   # inspect the parse IR
```

## License

MIT — see [LICENSE](LICENSE).
