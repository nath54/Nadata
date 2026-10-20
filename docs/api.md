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
```

`save_changes` returns the number of delta documents appended (1 when nothing
but the meta header is written). See [persistence.md](persistence.md).

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

## Error behaviour

| Situation | Raised |
|---|---|
| Unknown struct in `create_instance` | `KeyError` |
| Unknown symbol in `get` / `get_attribute` | `NameError` |
| Unresolvable identifier in executed code | `NameError` |
| Loading a non-Nadata `.ndb` file | `ValueError` |
| `save_changes` on a missing file | `ValueError` (use `dump` first) |
| Method call with undeclared argument | `TypeError` |
| Parameter or local shadowing an attribute | `SyntaxError` at resolve time |
