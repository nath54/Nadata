# Architecture

How a Nadata project becomes a running, persistent object graph. The engine
is a pipeline of six stages; each stage consumes the output of the previous
one and is independently testable.

```text
   .nastruct / .nadata text
            │
            ▼
   1. Tokenize & scan ─────►  raw tokens, comment-free spans
            ▼
   2. Parse ─────────────►   intermediate representation (TmpStruct, ASTs)
            ▼
   3. Resolve ───────────►   typed database (Struct, StructMethod, Types)
            ▼
   4. Load data ─────────►   instances, symbols, executed scripts
            ▼
   5. Runtime ───────────►   ExecutionScope evaluation, queries
            ▼
   6. Persist ───────────►   .ndb slotted-page file (dump / append / load)
```

All paths below are relative to `src/`.

---

## 1. Tokenizing and scanning

**Files:** `parser/expressions/tokenizer.py`, `parser/utils/`

The tokenizer turns source text into tokens: identifiers, keywords
(`class`, `fn`, `if`, `else`, `return`, `true`, `false`, `and`, `or`, `xor`,
`nand`, `not`), literals, punctuation, and operators. Comments are stripped by
a shared, offset-stable comment remover, so error positions stay accurate.

Two properties the rest of the engine relies on:

- The tokenizer **never advances past** the end of input (guarded).
- `->` is tokenized as an **operator**, which is what the lambda parser
  detects.

## 2. Parsing

**Files:** `parser/expressions/` (splitter, `expression_parser.py`,
`statement_parser.py`, `type_parser.py`, `dependencies.py`) and
`nadata/parser/nastruct_parser/`

Parsing happens in two layers:

1. **Expressions parse directly to final ASTs.** A precedence-climbing
   `ExpressionParser` builds nodes that implement `evaluate(scope) -> Value`.
   Statement parsing (`statement_parser.py`) produces typed declarations
   (`VarDeclStmt`), assignments, `if`/`else`, and `return`. Type annotations
   parse into `TmpType` trees (not yet resolved).
2. **Schema declarations parse into an intermediate representation.**
   `NaStructParser` (with the `parse_*` modules) produces `TmpStruct` /
   `TmpFunction` / `TmpAttribute` objects — types are still trees, bodies are
   still token spans. Nothing is registered in the database yet.

`.nadata` files are *classified* rather than fully pre-parsed:
`nadata/parser/nadata_loader/loader.py` recognizes instance declarations
(`symbol = Struct(...)`), assignments, and script statements. A declaration
whose right-hand side names a registered struct becomes an instantiation;
anything else is a plain script statement.

A debugging tool prints this intermediate representation:

```bash
python -m scripts.inspect_parse examples/ex0_simple --json
```

## 3. Resolution

**Files:** `nadata/parser/nastruct_parser/resolver.py`, `parser_state.py`

`SchemaResolver` converts the intermediate representation into the core
object model registered in a `NaDataDatabase`:

- **Structs** — `Struct` objects with stable type ids, resolved depth-first
  through parents. A struct is registered *before* its attributes resolve, so
  self references (`Array[Entity]` inside `Entity`) and forward references
  resolve correctly.
- **Types** — `TmpType` trees become real `Type` instances
  (`Int32`, `Array[T]`, `Dict[K, V]`, struct types, home types like
  `DateTime` and `Color`).
- **Defaults** — attribute defaults evaluate *eagerly* (once, at resolution);
  method parameter defaults are stored as expressions and evaluate *lazily*
  (per call, so `DateTime.now()` is fresh each call).
- **Bodies** — token spans become statement blocks; dependency analysis
  (`dependencies.py`) collects the free identifiers each body needs.
- **Shadowing check** — parameters or locals that share a name with an
  attribute (own or inherited) are rejected.

The resolver works on-demand through a `struct_provider` hook in
`TypeResolver`, which is what makes forward references across files work.
Because `NaStructParserState` accepts an existing database, resolution can
also target a live database (used by `Nadata.execute_schema`).

## 4. Loading data

**File:** `nadata/parser/nadata_loader/loader.py`

`NaDataLoader` executes `.nadata` content against a resolved database using
**three passes** — this is what makes forward and circular instance
references possible:

1. **Create** every declared instance as an empty shell and bind its symbol.
2. **Populate** constructor arguments in a scope where *all* symbols already
   exist — so `connected_peers=[edge_router]` works before `edge_router` is
   declared.
3. **Run scripts**: statements execute top to bottom, including method calls,
   function calls, `print`, and queries.

`StructRuntimeBinding` is registered per struct into the execution globals:
it acts as the constructor (`Book(...)`) and as the polymorphic query root
(`Book.where(...)`).

## 5. Runtime

**Files:** `nadata/core/expressions/` (`scope.py`, `expression.py`, `statement.py`),
`core/database/instance.py`, `core/database/queryset.py`

- **`ExecutionScope`** chains locals → parameters → globals → parent scopes
  and holds the receiver. The identifier `self` is bound to the receiver when
  one is in scope; there is no implicit attribute resolution (see
  [syntax.md](syntax.md) §6.3).
- **Expressions** evaluate to typed `Value` objects; `MemberAccessExpr`
  unwraps values, walks struct attributes (including inherited), and reaches
  native members of home types (e.g. `DateTime.year`).
- **`StructInstance`** stores attributes as name → `Value` and carries its
  `EntityId` (type id + local id). Methods execute with the receiver bound
  and parameters validated against the declaration.
- **`QuerySet`** provides `where` (expressions or Python callables), `count`,
  `first`, and chaining, over a struct and all of its descendants.

## 6. Persistence

**Files:** `core/database/persistence/`, `core/database/slotted_page.py`,
`core/database/page_manager.py`

A database persists to a single `.ndb` file — slotted pages holding framed
JSON documents. The full story, including the document kinds, the phased
loader, and incremental delta appends, is in
[persistence.md](persistence.md).

---

## Design invariants

- **No raw strings in the IR.** Everything is parsed into structured data
  (typed value objects, AST nodes, `Type` instances) before it reaches the
  core.
- **Two-phase everything.** Structs resolve shells-first for self reference;
  instances are created shells-first for forward reference. One-pass eager
  construction is impossible in a language with cycles.
- **Parser knows nothing about the database.** The parser produces IR; only
  the resolver registers objects. That is why the parser can feed both fresh
  databases and live ones.
- **Persistence is replayable.** The `.ndb` file is a log of documents
  processed in insertion order — which is why appends are just more
  documents.
