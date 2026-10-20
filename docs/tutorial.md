# Nadata Tutorial

A step-by-step introduction: you will model a small library, populate it,
query it, and persist it. Everything here runs with the commands shown — no
prior knowledge of the language needed.

Prerequisites: Nadata installed (`pip install -e .` from the repository root).

---

## Step 0 — Understand the two file types

A Nadata project is a folder with two kinds of source files:

| Extension | Contains |
|---|---|
| `.nastruct` | The **schema**: structs, attributes, methods, standalone functions |
| `.nadata` | The **data**: instances, symbols, script statements, queries |

Conventions used by all the bundled examples (and recommended for your own
projects):

```text
my_project/
    structs/
        *.nastruct
    data/
        *.nadata
```

The loaders pick up every `.nastruct` file under `structs/` and every
`.nadata` file under `data/`, in any subfolder. The separation is a
convention, not a hard rule — `Nadata.load` also accepts a single `.nastruct`
file.

---

## Step 1 — Your first schema

Create `library/structs/schema.nastruct`:

```text
class Book {
    title: Str;
    pages: Int32 = 0;
}
```

What this says:

- A struct `Book` with two attributes.
- `title` has no default: every `Book` must be given a `title`.
- `pages` defaults to `0`: it can be omitted.

Verify it parses:

```bash
nadata info library/structs/schema.nastruct
```

You should see a summary mentioning 1 struct and 0 instances. Schemas alone
are perfectly loadable — data is optional.

---

## Step 2 — Your first data

Create `library/data/books.nadata`:

```text
# Instances: symbol = Struct(attribute = value, ...)
hobbit = Book(title="The Hobbit", pages=310)
dune = Book(title="Dune")

# Script statements run top to bottom at load time
print("Loaded:", hobbit.title);
print("Dune pages default:", dune.pages);
```

> Note the style: `name="The Hobbit"` with a comma separator, optional
> trailing commas, and `;` after statements. Attribute-less names like
> `hobbit` become **symbols** — queryable handles for the instances.

Run the project:

```bash
nadata run library
```

Output:

```text
Loaded: The Hobbit
Dune pages default: 0
Loaded 1 struct(s), 2 instance(s), 0 function(s).
```

---

## Step 3 — Methods and `self`

Add a method to `Book` in the schema:

```text
class Book {
    title: Str;
    pages: Int32 = 0;

    @is_long() -> Bool {
        return self.pages > 300;
    }
}
```

Key rules for method bodies:

- **Explicit `self`**: instance state is *always* reached through `self.`
  (`self.pages`). A bare `pages` would be a `NameError` — parameters, locals,
  and global functions live in separate namespaces.
- **No shadowing**: a parameter or local may not share a name with an
  attribute (of the struct or any parent). The resolver rejects it.
- Locals are typed: `Int32 doubled = self.pages * 2;`
- Control flow: `if (condition) { ... } else { ... }`, booleans combine with
  keyword operators `and`, `or`, `not`.

Call it from data:

```text
print("Is the Hobbit long:", hobbit.is_long());
```

---

## Step 4 — Inheritance and polymorphism

```text
class Item {
    title: Str;

    @describe() -> Str {
        return self.title;
    }
}

class Book(Item) {
    pages: Int32 = 0;
    author: Str = "";
}

class Movie(Item) {
    minutes: Int32 = 0;
}
```

`Book` and `Movie` are both `Item`s. A query on the parent type returns
instances of all child types:

```text
dune = Book(title="Dune", author="F. Herbert");
arrival = Movie(title="Arrival", minutes=116);

# One query, two concrete types:
Array[Item] all_items = Item.where((item: Item) -> item.title != "");
print("Items in the library:", all_items.count());   # 2
```

Query predicates are **lambdas**: `(item: Item) -> expression`. The parameter
type annotation is required. Chain `.where(...)` calls to refine:

```text
Array[Book] long_books = Book
    .where((b: Book) -> b.pages > 300)
    .where((b: Book) -> b.author != "");
```

---

## Step 5 — Collections and references

Attributes can reference other instances and hold collections:

```text
class Shelf {
    label: Str;
    books: Array[Book] = [];
    metadata: Dict[Str, Str] = {};
}
```

Populate it — **order does not matter**, forward and circular references are
resolved in a second pass:

```text
fantasy = Shelf(
    label="Fantasy",
    books=[hobbit, dune],      # hobbit may be declared later!
    metadata={"floor": "1", "room": "east"}
)

hobbit = Book(title="The Hobbit", pages=310)
```

Dictionary literals are `{key: value}`, arrays are `[a, b, c]`, and both
allow trailing commas. Access members with `.`: `fantasy.books.count()`,
`fantasy.metadata` lookups, and indexed access on collections.

---

## Step 6 — Standalone functions

Reusable logic lives at top level with `fn`:

```text
fn heavier(a: Book, b: Book) -> Book {
    if (a.pages >= b.pages) {
        return a;
    }

    return b;
}
```

Functions are registered globally and callable from methods, scripts, and
query predicates — which is what makes queries composable:

```text
fn is_adult(h: Human) -> Bool { ... }

Array[Human] adults = Human.where((h: Human) -> is_adult(h));
```

---

## Step 7 — Persist and reload

```bash
# Save the whole database to a single file
nadata dump library store.ndb

# Inspect it
nadata info store.ndb

# Run it directly (scripts are stored as real ASTs and re-execute)
nadata run store.ndb
```

Incremental updates from Python — append only what changed:

```python
import nadata

app = nadata.Nadata.open("store.ndb")
app.execute_data('hitchhiker = Book(title="Hitchhiker", pages=224)')
written = app.save_changes("store.ndb")
print(f"appended {written} delta document(s)")

# Reopen: old and new data coexist
reopened = nadata.Nadata.open("store.ndb")
print(reopened.get_attribute("hitchhiker", "pages"))   # 224
```

---

## Step 8 — The Python API in one minute

```python
import nadata

app = nadata.Nadata.load("./examples/ex0_simple")

app.struct_names                 # ['Entity', 'Human', 'Animal', ...]
app.symbol_names                 # ['tom', 'jean', ...]
app.count("Human")               # 3

tom = app.get("tom")             # StructInstance
app.get_attribute("tom", "name") # 'Tom'
app.set_attribute("tom", "birthplace", "Lyon")

app.create_instance("Human", {"name": "Zoe"}, symbol="zoe")

app.call_method("tom", "older", app.get("jean"))
app.call_function("is_adult", app.get("tom"))

adults = app.query("Human").where(lambda h: h.get_attribute("name").get() == "Tom")
app.summary()
```

---

## Where to go next

- **[docs/syntax.md](syntax.md)** — the complete language reference, including
  every builtin type, operator precedence, and all the rules.
- **[docs/architecture.md](architecture.md)** — how your files become a
  running database.
- **[docs/api.md](api.md)** — full API and CLI reference.
- The six `examples/` projects — each one is a runnable lesson.
