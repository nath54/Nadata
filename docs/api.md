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
from src.console import NadataConsole

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
| Unknown symbol in `get` / `get_attribute` | `NameError` |
| Unresolvable identifier in executed code | `NameError` |
| Loading a non-Nadata `.ndb` file | `ValueError` |
| `save_changes` on a missing file | `ValueError` (use `dump` first) |
| Method call with undeclared argument | `TypeError` |
| Parameter or local shadowing an attribute | `SyntaxError` at resolve time |
