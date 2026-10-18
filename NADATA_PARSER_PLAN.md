# .nadata Parser — Implementation Plan

Everything needed to build the `.nadata` data-file parser, based on the current
state of the project (post resolution phase, commit `7d67344`).

---

## 1. What already exists (do NOT rebuild)

| Component | Location | Notes |
|---|---|---|
| Comment remover, bracket scanner | `src/parser/utils/` | Shared, offset-stable |
| Tokenizer | `src/parser/expressions/tokenizer.py` | `;` is punctuation |
| Token splitters | `src/parser/expressions/splitter.py` | `find_top_level`, `split_top_level` |
| Expression parser | `src/parser/expressions/expression_parser.py` | Full precedence; `parse_text`, `parse_expression_at` |
| Statement parser | `src/parser/expressions/statement_parser.py` | `parse_body(text) -> list[Statement]`; handles colon & space typed decls, assignments, if/else, return, expr statements |
| Type parser | `src/parser/expressions/type_parser.py` | `TmpType` trees |
| Dependency collection | `src/parser/expressions/dependencies.py` | AST walk, free identifiers only |
| Resolution phase | `src/parser/nastruct_parser/resolver.py` | `resolve_parser_state`, `build_type_globals`, `SchemaResolver` |
| Query layer | `src/core/database/queryset.py` | `where_expr(candidate_var, expr, scope)` WITH optimizer hoisting; `count()`, `first()`, chaining |
| Instance runtime | `src/core/database/instance.py`, `database.py` | `create_instance(struct, attributes, symbol)`, `set_attribute`, `call_method`, `symbols` dict |

**Key insight:** a `.nadata` file is a statement sequence. `StatementParser.parse_body`
already parses entire example data files: `tom = Human(...)` becomes an
`AssignStmt(CallExpr)`, `Array[Human] adults = ...` a `VarDeclStmt`, `print(...)`
an `ExprStmt`. The "parser" is mostly an *interpreter* over statements.

---

## 2. Gap 1 — Lambda expressions `(h: Human) -> expr`

The only missing syntax. Needed for `.where((h: Human) -> h.birthplace == "Paris")`.

- New AST node `LambdaExpr` in `src/core/expressions/expression.py`:
  - fields: `param_name: str`, `param_type_name: Optional[str]`, `body: Expression`
  - `evaluate(scope)` returns a **closure object** (new small class, e.g.
    `LambdaClosure` in the same module) holding `(param_name, body, defining_scope)`.
    The closure is callable: `__call__(instance)` binds the param via
    `scope.create_child_scope()` + `globals_dict[param_name] = instance`,
    evaluates body, returns `bool(result.get())`.
    Note: `QuerySet.where_expr`'s internal `eval_predicate` does exactly this
    pattern — mirror it.
- Parser support in `ExpressionParser._parse_primary` (or before it):
  detect `(` IDENTIFIER `:` type `)` `->` — parse param (reuse `parse_declaration_text`
  on the segment), then parse the body expression after `->`.
  Lambdas have no commas at top level, so argument splitting is unaffected.
- Type annotation of the param is cosmetic for now (resolution of param types
  in predicates is not enforced at runtime).

## 3. Gap 2 — Query entry point `Human.where(...)`

`MemberAccessExpr.get_callable` (expression.py) currently only handles Python
attributes and struct methods. Add: if the resolved target is a `Struct` (or a
`QuerySet`) and member is `where` / `count` / `first` / ...:

- `Struct` + `where` → return adapter building
  `QuerySet(struct=s, instances=table.all(), database=db, query_executor=...)`
  and returning `queryset.where(predicate)`.
- Problem: `CallExpr.evaluate` evaluates args BEFORE calling — a `LambdaExpr`
  arg evaluates to a `LambdaClosure` (callable) — that is fine: pass the closure
  to `where(predicate=closure)`; `where` accepts any callable.
- BUT the optimizer (`where_expr`) only kicks in through `where_expr`. For
  optimization, prefer the adapter to detect `LambdaClosure` args and route to
  `where_expr(candidate_var=closure.param_name, expr=closure.body, scope=closure.scope)`.
  Optional but nice (it is the documented selling point).
- `Human` must resolve in globals to the **Struct** object (see loader globals).

## 4. Gap 3 — The loader `src/parser/nadata_parser/loader.py`

New package `src/parser/nadata_parser/` with `loader.py` (+ `__init__.py`).

### 4.1 Globals

```
globals = build_type_globals()                     # DateTime, Color
+ {"print": print}                                 # builtin
+ {struct_name: Struct object}                     # for every registered table
  (constructors + .where entry point both come from the Struct)
```

Constructor calls (`ObjectInstance(knife)`) resolve via `CallExpr._resolve_callable`:
`IdentifierExpr("ObjectInstance")` → symbol is a Struct (has no `execute`) →
returned as-is → not callable → **CallExpr raises TypeError today.**
→ Add a `StructConstructor` callable wrapper (or make `CallExpr` handle Structs):
`StructConstructor(struct, database)` with `__call__(self, *args, **kwargs)`
creating an instance (kwargs → set_attribute after creation with defaults).

### 4.2 Two-phase instantiation (forward / circular references)

ex2 data: `db_primary` references `edge_router` / `replica_backup` before their
declarations, and the references are **circular**. Eager one-pass creation is
impossible → two phases:

1. **Pass 1 — declare**: walk top-level statements; for every
   `AssignStmt` whose `value_expr` is `CallExpr` with `callee = IdentifierExpr(S)`
   where `S` is a registered struct name:
   - `instance = database.create_instance(struct=tables_by_struct_name[S].struct, symbol=name)`
   - record `(symbol, instance, value_expr)` in a pending list.
2. **Pass 2 — populate**: for each pending item, build an `ExecutionScope` whose
   `globals_dict` = loader globals **+ all instance symbols** (name → StructInstance),
   evaluate the CallExpr kwargs (arrays of symbols resolve to the instances),
   then for each kwarg: `instance.set_attribute(name, value)`.
   Nested constructor calls (`ObjectInstance(knife)`) evaluate normally here
   and create anonymous instances.
3. Validation: `set_attribute` runs `attr_type.check_value` — wrong types fail
   loudly. `collect_dependencies` can pre-validate that every referenced
   symbol is either a struct type, builtin, or a declared instance.

### 4.3 Script execution

Pass 3 — for every remaining top-level statement (`ExprStmt`, `VarDeclStmt`,
non-instance `AssignStmt`), execute via `Statement.execute(scope)` /
`expression.evaluate(scope)` in a scope with the loader globals **+ instance
symbols** (so `is_adult(tom)` and `tom.describe()` resolve).

- `VarDeclStmt` for `Array[Human] adults = Human.where(...)`: init expr evaluates
  to a QuerySet → `scope.define` stores it; later `adults.where(...)` works
  because `QuerySet` has a real `where` method (native path already works).
- Note: `QuerySet` methods are real Python methods → `MemberAccessExpr.get_callable`
  native path handles `.count()`, chained `.where()` on results, etc. Only the
  `Struct` → QuerySet entry needs the adapter (gap 2).

### 4.4 API

```
class NaDataLoader:
    def __init__(self, database: NaDataDatabase) -> None
    def load_file(self, content: str) -> None      # parse + 3 passes
    def load_folder(self, folder_path: str) -> None  # all .nadata files, shared scope
```

Consider a facade `NaDataProject.load(structs_folder, data_folder)` that runs
NaStructParser.parse_folder then the loader.

---

## 5. Test plan (`tests/test_nadata_loader.py`)

1. ex0 data: after load, `db.symbols` contains tom/jean/louise/white_cat/knife;
   `tom.get_attribute('name').get() == 'Tom'`; `white_cat` is a Cat instance;
   `tom.get_attribute('parents')` returns jean & louise instances.
2. Circular refs (ex2 data): `db_primary.get_attribute('connected_peers')`
   contains replica_backup AND vice versa.
3. Queries: build scope, evaluate `Human.where((h: Human) -> h.name == "Tom").count()`.
4. Script statements: `print` executes, typed decls store QuerySets.
5. Wrong attribute type on set_attribute raises TypeError.
6. Unknown symbol in kwargs raises a clear error.

## 6. Acceptance criteria

`examples/ex0_simple` and `examples/ex2_inheritance` (structs + data) load and
execute fully, printing the expected output of their `print(...)` statements
(especially ex2: `all_nodes.count()` polymorphic query, `primaries`,
`fast_heavy_dbs.count()`).

## 7. Open decisions (ask the owner)

1. `print` builtin: plain Python print OK, or a Nadata value formatter?
2. Anonymous nested instances (`ObjectInstance(knife)`): no symbol binding — OK?
3. Lambda param type (`(h: Human) ->`): enforced at runtime or cosmetic?
4. Should `load_folder` share one scope across files (cross-file symbols)?
   (Recommended: yes.)
