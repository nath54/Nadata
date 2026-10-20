# Nadata

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
  methods and computed properties, standalone functions
- **Rich type system** — sized integers/floats (`Int32`, `UFloat16`, ...),
  `Str`, `Bool`, `DateTime`, `Color`, `Array[T]`, `Dict[K, V]`, entity
  references
- **Declarative data files** — forward and circular references, script
  statements, `print`, chained polymorphic queries
- **Persistence** — whole-database dumps and incremental delta appends to a
  single `.ndb` file
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
| [docs/development.md](docs/development.md) | Code style, testing, contributing |

## Examples

The `examples/` folder contains six self-contained projects, from basics to
edge cases:

| Example | Demonstrates |
|---|---|
| `ex0_simple` | Family graph: inheritance, methods, forward refs, queries |
| `ex1_primitives` | Every sized integer/float type and boundary values |
| `ex2_inheritance` | Circular references, polymorphic queries |
| `ex3_collections` | Nested arrays, dictionaries, `Color`, `DateTime` math |
| `ex4_expressions` | `if`/`else`, boolean operators, local declarations |
| `ex5_edge_cases` | Parser edge cases: comments, spacing, empty constructs |

## Development

```bash
python -m pytest tests/ -q        # run the test suite
python -m mypy src/               # type check (strict)
python -m scripts.inspect_parse examples/ex0_simple   # inspect the parse IR
```

## License

See the repository for license information.
