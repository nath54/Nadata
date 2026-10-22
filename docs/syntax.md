# Nadata Language Syntax Specification

This document specifies the exact syntax of the two Nadata source file formats:

- **`.nastruct`** — schema files: struct (class) definitions and standalone functions.
- **`.nadata`** — data files: instance declarations, named symbols, and script statements.

New to Nadata? Start with the [tutorial](tutorial.md); for the engine internals
see [architecture.md](architecture.md).

The reference examples live in `examples/` (`ex0_simple` to `ex9_constants`) and are
the canonical acceptance material for the parsers.

---

## 1. General rules

| Rule | Value |
|---|---|
| Encoding | UTF-8 |
| Case sensitivity | Fully case-sensitive (`Human` ≠ `human`) |
| Whitespace | Spaces, tabs and newlines are insignificant except inside strings and comments |
| Identifiers | `[A-Za-z_][A-Za-z0-9_]*` (letters, digits, underscore; cannot start with a digit) |
| Keywords | `class`, `fn`, `if`, `elif`, `else`, `return`, `true`, `false`, `and`, `or`, `xor`, `nand` |

---

## 2. Comments

Comments follow Python conventions. Two forms exist:

```text
# Line comment: from '#' to the end of the line.

""" Block comment:
    may span any number of lines,
    ends at the next '"""' on its own or embedded in a line. """
```

Rules:

- `#` comments end at the newline character. The newline itself is **not** part of the comment.
- `"""` block comments may span lines and may appear anywhere whitespace is allowed.
- Comment markers (`#`, `"""`) appearing **inside a string literal** are plain characters
  and are never treated as comments.
- Inline comments are allowed everywhere: `y: Int32; # Trailing comment after semicolon`.

**Greedy scanning rule.** Scanning is strictly left-to-right. A string literal cannot
contain `"""`. The sequence `"abc""" this"` is parsed as: string `"abc"`, then empty
string `""`, then ` this` starts an unterminated string (error). To embed `"""` in
text, use escaped quotes (see below).

---

## 3. String literals

- Strings are delimited by **double quotes `"..."` only**.
- **Single quotes `'...'` are NOT valid string delimiters** and are reserved for
  future use. `'abc'` is a syntax error.
- Strings are single-line in principle; a raw newline inside a string literal is a
  syntax error (use escapes or concatenation).
- Escape sequences, introduced by backslash `\`:

| Escape | Meaning |
|---|---|
| `\"` | Literal double quote |
| `\\` | Literal backslash |
| `\n` | Newline |
| `\t` | Tab |

- An escape followed by any other character is an error (strict) or the literal
  character (lenient — to be decided once the parser reaches string handling).
- A backslash escaping a quote does **not** terminate the string: the scanner counts
  the consecutive backslashes preceding a `"` — an odd count means the quote is escaped.

Examples (from `ex5_edge_cases`):

```text
str_with_symbols = "String containing quotes: \"escaped\" and brackets [ { ( ) } ]"
empty_str = ""
```

---

## 4. Numbers and booleans

- **Integers**: decimal digits, optional leading `-`. No hex/octal/binary literals yet.
- **Floats**: decimal notation with a fractional part (`21.5`, `0.90`) and
  scientific notation (`1e6`, `3.5e-2`, `1.5E+3`). Exponents require at least one
  digit; `1e` is not a float.
- **Booleans**: the keywords `true` and `false` (lowercase only).
- Unary minus is an operator, not part of the literal: `x = -0`, `tiny_val = -128`.

---

## 5. Types

### 5.1 Primitive types

| Type | Description |
|---|---|
| `Str` | Unicode string |
| `Bool` | Boolean |
| `Int2`, `Int4`, `Int8`, `Int16`, `Int32`, `Int64` | Signed sized integers |
| `UInt2`, `UInt4`, `UInt8`, `UInt16`, `UInt32`, `UInt64` | Unsigned sized integers |
| `Float8_E4M3`, `Float8_E5M2`, `Float16`, `BFloat16`, `Float32`, `Float64` | Signed floats (total bits / exponent bits) |
| `UFloat8_E4M3`, `UFloat8_E5M2`, `UFloat16`, `UBFloat16`, `UFloat32`, `UFloat64` | Unsigned floats |

### 5.2 Generic types

```text
Array[T]              typed array, e.g. Array[Str], Array[Entity]
Array[Array[T]]       nested generics, e.g. Array[Array[Float32]]
Dict[K, V]            typed dictionary, e.g. Dict[Str, Str], Dict[Str, Array[Str]]
Set[T]                typed set of unique elements, e.g. Set[Str], Set[Entity]
Queue[T]              typed FIFO queue, e.g. Queue[Str], Queue[Task]
PriorityQueue[T]      typed priority queue (Int64 priorities), e.g. PriorityQueue[Task]
PriorityQueue[T, P]   typed priority queue with basic numeric priorities P, e.g. PriorityQueue[Task, Int32]
```

### 5.3 Sets

```text
Set[Str] tags = {"admin", "dev", "admin"};   // deduplicates to {"admin", "dev"}
Set[Str] empty = {};                          // empty braces declare an empty set
if ("admin" in p.tags) { ... }                // O(1)-style native membership probe
for (t: Str in p.tags) { ... }                // iteration (unordered)
```

- Set literals use braces with comma separated elements: `{1, 2, 3}`;
  a single element `{x}` is a one-element set. **Empty braces `{}` are a
  dict** (Python convention) — write an empty set through a declared
  `Set[T]` type.
- Elements must match the declared element type and be **hashable**
  (scalars, strings, home types, struct instances — identity semantics
  for instances). Nested containers (`Set[Array[Int32]]`) are not
  representable.
- Sets persist in the binary format with elements written in sorted
  order, so the same logical set always encodes to the same bytes.

### 5.4 Queues

```text
Queue[Str] jobs = {};                 // FIFO queue, filled with push
jobs.push("repair");                  // enqueue at the back
head = jobs.peek();                   // front element (not removed)
next = jobs.pop();                    // serve the front element
n = jobs.size();                      // element count

PriorityQueue[Task] pq = {};          // priorities default to Int64
PriorityQueue[Task, Int32] q32 = {};  // explicit basic numeric priority
pq.push(task, 3);                     // insert with priority
urgent = pq.pop();                    // serve the LOWEST priority first
```

- **Priorities are restricted to basic `IntXX` / `FloatXX`** (including
  unsigned variants); any other priority type is a resolution error.
- `pop` / `peek` on an empty queue raise; `push` validates the element
  and the priority against the declared types.
- Elements must be hashable-free but *storable*: scalars, strings, home
  types, and struct instances are supported; nested containers are not.
- Queues persist in the binary format; priority queues are written in
  normalized (priority, element) order so the same content always
  encodes to the same bytes.
- Iteration (`for x in q`) yields elements in serving order; `in`
  probes elements.

### 5.5 Sparse spatial grids

Sparse hash-based grids storing only occupied cells. Coordinates are
fixed `Int64` and negative values are allowed; three topologies are
provided:

```text
GridSquare2D[T]   2D square grid, 4-adjacency (up/down/left/right)
GridHex2D[T]      2D hexagonal grid, axial coordinates (q, r), 6-adjacency
GridCube3D[T]     3D cube grid, 6-adjacency (face neighbors)
```

```text
GridSquare2D[Str] terrain = {};       // empty sparse grid
terrain.set_at(0, 0, "grass");        // write a cell (coords validated)
t = terrain.at(-3, 7);                // read a cell (missing raises)
b = terrain.has(5, 5);                // occupancy probe
n = terrain.size();                   // occupied cell count
terrain.remove(1, 0);                 // delete a cell
coords = terrain.coords();            // Array[Array[Int64]], sorted
nb = terrain.neighbors(0, 0);         // all neighbor coordinates
occ = terrain.occupied_neighbors(0, 0);  // only existing cells
for (c: Str in terrain) { ... }       // iterate occupied values, coord-sorted
```

- All members are topology aware: `neighbors` on `GridHex2D` uses the
  six axial directions, on `GridCube3D` the six face directions.
- Only occupied cells are stored, so huge sparse worlds cost nothing.
- Grids persist in the binary format with cells written in sorted
  coordinate order (zigzag varints, negatives included), so the same
  content always encodes to the same bytes.

### 5.6 Dense tensors

Dense n-dimensional tensors with the **shape declared in the type** and
checked on every write:

```text
Tensor[Float32, 3, 4] m;              // 3x4 matrix, 12 elements
t = m.at(0, 2);                       // bounds-checked element read
m.set_at(1, 3, 0.5);                  // element write (flat index 7)
x = m[1, 3];                           // multi-index read (per-dim)
m[0, 1] = 0.5;                         // multi-index write
row = m[1, :];                         // slice: whole second row
block = m[0:2, 1:3];                   // slice: 2x2 sub-block (copy)
m[0:2, 1] = [0.1, 0.2];                // block write (row-major payload)
m[:, 0] = 0.0;                         // block write (scalar broadcast)
s = m.shape();                         // Array[Int32] of dimensions
n = m.size();                          // total element count
x = m.sum();                           // elementwise reductions: sum/min/max
m.fill(0.0);                           // in-place fill of every cell
t = m.transpose();                     // square 2-D transpose (copy)
for (x: Float32 in m) { ... }         // row-major iteration
"x" in m                               // membership probe (flat values)
```

**Initializers** — `zeros()`, `fill(v)`, and `eye()` materialize
against the target's declared tensor type, so they work as schema
defaults, constructor arguments, and typed locals:

```text
m: Tensor[Float32, 3, 4] = zeros();    // all 0.0
w: Tensor[Float32, dim, dim] = eye();  // identity (symbolic dims OK)
z: Tensor[Float32, 2, 3] = fill(2.5);  // all 2.5
```

- `zeros()` and `eye()` require numeric element types; `eye()`
  requires a square 2-D shape. `fill(v)` accepts any element type.
- Initializers under symbolic dimensions resolve per instance from
  the bound constants (so they survive persistence round-trips).
- Using an initializer anywhere other than a declared tensor target
  is a `TypeError`.

**Elementwise arithmetic** — `+`, `-`, `*`, `/` work between
tensors of identical shape (elementwise) and between a tensor and a
numeric scalar (broadcast); results keep the base tensor's shape and
element type. Division on integer element types is a `TypeError`
(the result would be fractional).

**Value-wise function application** — `map` applies a function to
every cell. The function is a lambda or a named standalone function
passed by identifier; results must satisfy the element type:

```text
y = x.map((v: Float32) -> v * 2.0);   // lambda
y = x.map(relu);                      // named function (fn declared in schema)
soft = z.map((v: Float32) -> exp(v)); // compose with the math builtins
soft = soft / soft.sum();             // e.g. a softmax
```

**Math builtins** — `abs(v)`, `exp(v)`, `sqrt(v)`, `log(v)`, and
`pow(base, exponent)` operate on numeric values (and through `map`
on every tensor cell).

**Convolutions and pooling** — deep-learning style valid
cross-correlation (no kernel flip, no padding), on 1-D and 2-D
numeric tensors:

```text
y = x.conv1d(kernel);            // kernel: Tensor[T, K]; out length (L-K)//stride+1
y = x.conv1d(kernel, 2);         // optional stride (default 1)
y = img.conv2d(kernel);          // kernel: Tensor[T, kH, kW]; out (OH, OW)
y = sig.max_pool(2);             // window 2, stride defaults to the window
y = sig.min_pool(2, 1);          // optional stride
y = img.avg_pool(2);             // average pooling (float element types only)
```

- Kernels are passed as tensors; the output shape is
  `(axis - kernel) // stride + 1` per axis, and a kernel or window
  larger than the input axis is a `TypeError`.
- `conv1d` expects a 1-D tensor and 1-D kernel; `conv2d` a 2-D tensor
  and 2-D kernel; pooling accepts rank 1 or 2.
- `avg_pool` requires a float element type (the average is
  fractional); `max_pool`/`min_pool` also accept integers.

**Slicing** — each index position may be an integer (drops the
dimension, returning an element) or a `start:stop` slice (keeps the
dimension, returning a new copied tensor):

- Bounds are optional: `[:2]` starts at 0, `[1:]` runs to the end,
  `[:]` selects the whole axis; both bounds accept arbitrary integer
  expressions.
- Bounds are strict: `0 <= start <= stop <= dim` (no negatives, no
  steps); an empty or reversed slice is a `ValueError`, an integer
  index out of range is an `IndexError`.
- The result of a slice is an **independent copy** with the sliced
  shape (`m[0:2, 1:3]` of a 3x4 is a 2x2 `Tensor[Float32, 2, 2]`).
- Slice **assignment** takes a flat or nested array literal whose
  row-major length matches the sliced shape, or a scalar that is
  broadcast to every selected cell; elements are type-checked against
  the tensor's element type.

Method parameters may use symbolic dims, validated against the
receiver's constants at call time:

```text
@forward(x: Tensor[Float32, batch, dim]) -> Float32 { ... }
```

Symbolic dims are not allowed on standalone function parameters or on
return types. Arrays and strings expose `.length()`; typed containers
(`Dict`, `Set`, queues) expose `.size()`.

- Dimensions must be positive integer literals in the type, **or the
  names of `const` integer attributes of the struct** (symbolic dims,
  see 5.7); nested array literals flatten row-major on assignment, and
  the flattened length must equal the product of the dimensions.
- Tensors persist in the binary format as a varint count followed by
  the flat row-major elements (the shape lives in the schema).

### 5.7 Constants and symbolic dimensions

A `const` attribute binds once — in the constructor — and is immutable
afterwards:

```text
class Attention {
    const batch: Int32;              // required: no default
    const dim: Int32 = 4;            // optional: schema default
    q: Tensor[Float32, batch, dim];  // shape refers to the constants
}

a = Attention(batch = 2, q = ...);   // constructor binding only
```

- Constants bind through constructor arguments (or their schema
  default); every later write path — `set_attribute`, target
  assignment, `self.X = ...` in methods — is a type error.
- Tensor dimensions may name `const` attributes of the same struct
  (inherited ones included); the referenced attribute must be a basic
  integer type. Declaration order inside the body is irrelevant.
- Tensor payloads are validated at construction against the bound
  extents; two instances of the same struct may bind different values
  (`batch = 2` and `batch = 3` coexist).
- A tensor with symbolic dims cannot be written until all its named
  constants are bound. Single-cell writes (`set_at`) never re-check
  the shape.
- A const default may **derive from other constants** by referencing
  them through `self`:

  ```text
  const half: Int32 = self.dim / 2;
  const fourdim: Int32 = self.dim * 4;
  ```

  Derived constants are computed per instance at construction, in
  declaration order, and cannot be set directly. Referenced attributes
  must be consts; an integral float result coerces to the declared
  integer type (a remainder, e.g. `dim = 5` with `/ 2`, is a type
  error).
- Constants persist as ordinary attributes; the binary format is
  unchanged.

### 5.8 Home types

| Type | Construction |
|---|---|
| `DateTime` | `DateTime(y=2003, m=4, d=12)`, `DateTime()` (epoch zero), `DateTime.now()` |
| `Color` | `Color(r, g, b)`, `Color(r, g, b, a)` (channels 0–255) |

`DateTime` field keywords: `y`, `m`, `d`, `H`, `M`, `S`, `MS`.

### 5.9 Struct types

Any struct declared in a `.nastruct` file is itself a type, usable as attribute type,
array element, dict value, parameter type, or return type:

```text
parents: Array[Entity] = [];
object_type: ObjectType;
```

---

## 6. `.nastruct` files — schemas

A `.nastruct` file is a sequence of top-level blocks: `class` definitions and
standalone `fn` definitions, separated by whitespace and comments.

### 6.1 Class declaration

```text
class Name { ... }
class Name(Parent) { ... }      // single inheritance
class Name {}                   // empty body allowed (on same line too)
class ChildOfEmpty(EmptyClass){}
```

- Inheritance is **single** parent, declared in parentheses after the class name.
- A class body may be empty: `{}`.
- Body delimiters are curly braces `{ }` — always braced, no brace-less forms.

### 6.2 Attribute declarations

```text
name: Type;                     // required attribute, no default
name: Type = default;           // attribute with default value
```

- Attributes end with a semicolon `;`.
- The default value must type-check against the declared type.
- **Naming rule**: attribute names, method parameter names, and local variables
  must be `lower_snake_case` — they must not start with an uppercase letter.
  UpperCamelCase is reserved for type names (structs and builtins such as
  `Int32`, `Color`, `GridHex2D`). Underscore-prefixed names (`_count`) are
  allowed. Violations are a `SyntaxError` at resolution time (or, for `.nadata`
  script variables, when the statement executes).
- Allowed default expressions: literals (`"Sensor_A"`, `21.5`, `-128`, `true`, `""`),
  constructor calls (`Color(0, 0, 0)`, `DateTime()`), function-like calls
  (`DateTime.now()`), and empty collections (`[]`, `{}`).
- Attributes declared in a parent are inherited; redeclaring the same name in a
  child is an error.

### 6.3 Methods

```text
@name(param: Type, param2: Type = default) -> ReturnType {
    ...body...
}
```

- Method declarations are prefixed with `@`.
- Parameters are `name: Type`, comma-separated; trailing commas allowed.
- Parameters may have default expressions (evaluated **at each call**, so
  `now: DateTime = DateTime.now()` is refreshed per call).
- **Parameter order is free**: parameters with default values may be declared
  before, after, or between required parameters. Positional call binding follows
  the declared order, named binding follows names — so no ordering constraint
  is needed.
- The return type is mandatory: `-> Type`.
- The body uses the same statement language as `.nadata` scripts (see §7.5).

#### Explicit `self`

Attribute and method access on the receiver is **always explicit** through
`self`:

```text
@is_broken() -> Bool {
    broken: Bool = false;
    if (self.state <= 0) {
        broken = true;
    }
    return broken;
}
```

- `self` is a read-only identifier bound to the receiver instance; it is only
  defined inside method / computed property bodies (using it in a `.nadata`
  script is a `NameError`).
- Bare attribute names do **not** resolve: `return uid;` is a `NameError`, use
  `return self.uid;`. This keeps parameters, locals, attributes, and global
  functions in disjoint namespaces.
- **Shadowing is a resolution error**: a method parameter or local variable may
  not share a name with an attribute of the struct or any of its parents.
  This applies transitively to inherited attributes.
- `self` may be used as a value: passed to functions (`self == other`), stored,
  or returned.
- Calling another method on the receiver: `self.age(now)`.

### 6.4 Computed properties

A method with an **empty parameter list** is a computed property:

```text
@is_broken() -> Bool {
    broken: Bool = false;
    if (self.state <= 0) {
        broken = true;
    }
    return broken;
}
```

It is called exactly like any method, **with parentheses**: `item.is_broken()`.

Rules:

- The parameter list `(...)` is **mandatory** in every `@` declaration, even when
  empty: `@name -> Type { ... }` without parentheses is a syntax error.
- The parentheses are **mandatory at every call site**, even for zero-argument
  methods and computed properties: `item.is_broken()`, never `item.is_broken`.
- There is no syntactic distinction at call time between methods and computed
  properties — both are `name(...)`.

### 6.5 Standalone functions

```text
fn name(h1: Human, now: DateTime = DateTime.now()) -> DateTime {
    DateTime age1 = h1.age(now);
    return age1;
}
```

- Declared at top level with the `fn` keyword.
- Same parameter / return type rules as methods (see §6.3).
- Registered globally in the database; callable from methods and scripts.

### 6.6 Statements allowed in method / function bodies

```text
Float32 scaled = x * self.scale_factor;   // typed variable declaration
broken = true;                            // assignment
if (condition) { ... } elif (cond2) { ... } else { ... }   // conditionals
return result;                            // return (mandatory last expression)
```

- Conditions are parenthesized: `if ((flag_a and not flag_b) or flag_c) { ... }`.
- `elif` chains (any count) and a final `else` are allowed; `else if (...)`
  is accepted as an alias for `elif (...)`.
- Operators: `+ - * / %`, comparisons `== != < <= > >=`,
  logic (keyword only): `and`, `or`, `xor`, `nand`,
  membership (keyword only): `in`, `not in`.
  - `+` on two `Str` operands is **string concatenation**; mixing `Str` with
    any other operand kind is a resolution error (strict typing).
- **Membership** (`in` / `not in`) binds like a comparison and works on:
  - `Array` — element scan: `"vip" in p.tags`,
  - `Dict` — **key** probe (hash lookup): `"floor" in config`,
  - `Set` — element probe (hash lookup): `"vip" in p.tags`,
  - `Str` — substring search: `"lo wo" in s`.

  These are single native containment checks — no interpreted loop, no
  loop fuel. The needle must fit the container (`"x" in int_array` and
  `5 in "abc"` are type errors). Chaining (`a in b in c`, `a < b in c`)
  is a syntax error; combine with `and`/`or` instead.
- Local variable declarations may be typed (`Float32 result = scaled + self.bias;`).
- Member access with `.`; method calls with `()`.
- **Indexing** with postfix `[...]` on arrays and dictionaries (chainable and
  freely mixed with member access and calls):

  ```text
  tom.parents[0].name            // array index, then member access
  matrix[0][1]                   // chained index into nested arrays
  config["floor"]                // dictionary index by key
  tags[index + 1]                // index may be any expression
  ```

  - Array indices are integers; Python semantics apply for negatives
    (`[-1]` is the last element). Out-of-range raises `IndexError`.
  - Dictionary keys are validated against the declared key type; assigning
    to a new key inserts it, assigning to a wrong-typed key raises
    `TypeError`.
  - **Index and member assignment** is supported: `arr[0] = x`,
    `config["floor"] = v`, `self.count = n`, and nested chains such as
    `self.children[0].name = s`. Array elements and dict values are
    validated against their declared element types on every write.
- Instance state is reached only through `self.` (see §6.3): bare names resolve
  to parameters, locals, and global functions — never attributes.
- **Compound assignments and increments** (statements, not expressions):
  `x += e`, `x -= e`, `x *= e`, `x /= e`, `x %= e`, `x++`, `x--`. Each
  desugars into a plain assignment over the existing binary operator
  (`x += e` is `x = x + e`), so the strict operand rules of §6.6 apply
  (`s += "!"` concatenates; `x++` requires Int/Float). Compound forms target
  plain identifiers, members (`self.count += 1`), and indices (`arr[i]++`,
  `self.items[0] += 1`) — the target is re-read as the left operand.
- **Loops** (see §6.8): `for (...) { ... }`, `while (...) { ... }`, with
  `break` / `continue` confined to loop bodies (syntax error outside).

### 6.8 Loops

All loops are **fuel-guarded**: each loop may run at most `loop_limit`
iterations (default **10,000**), then a `LoopLimitError` aborts execution.
The limit is a database setting — change it with the native
`set_loop_limit(n)` from script code or `app.set_loop_limit(n)` from Python;
the value persists in the `.ndb` meta document.

**For-in** — iterate an array's elements (or a dict's keys / values). The
loop variable is **always typed** and scoped to the loop body:

```text
for(x: Int32 in numbers) { total = total + x; }
for(h: Human in adults) { print(h.name); }
for(k: Str in config.keys()) { ... }
for(v: Int32 in config.values()) { ... }
```

- Each element is checked against the declared type as it is bound
  (subtypes pass; a mismatch raises `TypeError` naming the element).
- Only arrays are iterable; `for(k in d)` (direct dict iteration) is
  rejected as ambiguous — use `.keys()` / `.values()`.
- The loop variable must be **fresh**: it may not shadow a parameter,
  attribute, or local visible at the loop (resolution error). Sequential
  loops may reuse a name, since loop variables go out of scope.

**While**:

```text
while(count < 10) { count++; }
```

**C-style** — sections separated by `::` (commas stay free for the
comma-separated init / update lists). All three sections are mandatory and
every init declaration is typed with a mandatory initializer:

```text
for(i: Int32 = 0 :: i < 10 :: i++) { ... }
for(x: Int32 = 0, y: Int32 = 10 :: x + y < 100 :: x += 1, y *= 2) { ... }
```

- Init declarations and the loop variable live in the loop scope and vanish
  after the loop (same freshness rule as for-in).
- Update entries may be assignments, compound assignments, or `++` / `--`.
- `break` exits the innermost loop; `continue` skips to its next iteration.

### 6.7 Built-in members and functions

A small set of built-ins is always available in method bodies and scripts.

**Entity id** — every instance exposes its per-struct instance index:

```text
print(tom.id);                    // 0 (the index shown in <Human id=0>)
```

**`type()`** — returns the schema type of its argument, as a type handle:

```text
print(type(tom));                 // <Type Human>
print(type(42));                  // <Type Int64>
print(type(Human.birthday));      // <Type DateTime>
print(type(Human.pages));         // <Type Int32>
```

**Member access on a struct type** — `Struct.attribute` is itself the
attribute's declared type (see the last two lines above).

**Introspection members** — available on instances *and* on struct types:

```text
tom.list_attr();                  // ['name', 'gender', ..., 'size', ...]
Human.list_attr();                // same, from the schema side
Human.list_methods();             // ['age', 'older'] (inherited included)
Human.getattr("size");            // <Type UFloat16> (on a type: the declared type)
tom.getattr("birthplace");        // current value of the attribute
tom.setattr("birthplace", "Lyon"); // schema-constrained write
```

- `getattr` / `setattr` are **constrained to the class schema**: the name must
  be a declared attribute of the struct or one of its parents, otherwise an
  `AttributeError` is raised. `setattr` additionally validates the value
  against the declared attribute type.
- `list_attr()` / `list_methods()` include everything inherited.

**Schema introspection** — database-wide, callable from scripts and bodies:

```text
print(schema_classes());          // [<Type Entity>, <Type Human>, ...]
print(schema_functions());        // [calculate_age_difference, is_adult, ...]
```

---

## 7. `.nadata` files — data and scripts

A `.nadata` file mixes three kinds of content: **instance declarations**,
**script statements**, and **queries**.

### 7.1 Call arguments: named or positional, never mixed

Every call site (methods, functions, constructors, computed properties) accepts
exactly one calling style:

```text
obj.method(value1, value2)               // positional, declaration order
obj.method(param1=value1, param2=value2) // named
obj.method()                             // zero args: parentheses still mandatory
```

- **Mixing positional and named arguments in the same call is an error.**
- With named arguments, any parameter order is allowed.
- With positional arguments, values bind in declaration order.
- Trailing separators are allowed in argument lists.

### 7.2 Instance declarations (named symbols)

```text
symbol_name = ClassName(
    attr=value,
    attr2=[sym_a, sym_b],
    attr3={ "key": value },
)
```

- `symbol_name` becomes a database symbol; later declarations can reference it.
- **Forward references are allowed**: an instance may reference symbols declared
  later in the file (including cyclic references).
- Arguments follow the common call rules (§7.1), comma-separated;
  **trailing commas allowed** (both in argument lists and arrays).
- Empty instantiation is valid: `item2 = EdgeCaseItem()` (all defaults apply).
- Single-line form is valid: `item3 = EdgeCaseItem(x=10, y=20)`.
- Values may be: literals, arrays `[a, b, c]`, dictionaries `{ "k": v }`,
  symbol references, nested constructor calls, negative numbers.

### 7.3 Anonymous inline instances

An instance may be constructed inline, without a symbol, inside another value
(e.g. inside an array or a `Dict` value).

### 7.4 Typed script declarations

```text
Float32 test_input = 10.0;
Array[MathPipeline] heavy_pipes = MathPipeline.where(...);
```

- Syntax: `Type name = expression;`
- The declared `Type` must match the expression's evaluated type.

### 7.5 Plain script statements

- Assignment: `name = expression;`
- Expression statements (calls): `pipe_standard.apply_transform(test_input);`
- Built-in `print(...)`: `print("Label:", value);` — accepts multiple arguments
  separated by commas.
- Semicolons terminate statements; when the statement ends at end of line, the
  semicolon is **optional** (both styles appear in the examples).

### 7.6 Queries

```text
Array[Node] all_nodes = Node.where((n: Node) -> n.uid != "");
Array[DatabaseServer] primaries = DatabaseServer.where(
    (ds: DatabaseServer) -> ds.is_primary == true
);
```

- Syntax: `StructName.where((var: StructName) -> boolean_expression)`
- Returns a queryable collection: `.count()`, `.first()`, `.last()`, `.exists()`,
  chained `.where(...)`, `.limit(n)`, `.to_list()`, and iteration via for-in.
- **Ordering**: `StructName.order_by("age")` (ascending) or
  `StructName.order_by("age", descending = true)`. The sort is stable.
- **Aggregation**: `StructName.sum_of("age")`, `.min_of(...)`, `.max_of(...)`,
  `.average_of(...)` — numeric attributes only; `StructName.pluck("name")`
  extracts raw payloads.
- All of the above are also available on query results:
  `Person.where(...).order_by("age").limit(5)`.
- The lambda parameter shadows struct members of the same name.
- The query engine pre-evaluates loop-invariant sub-expressions once
  (e.g. `now` in `h.age(now)` is computed a single time, not per candidate).

---

## 8. Punctuation summary

| Token | Usage |
|---|---|
| `{ }` | Class / function bodies, block comments (`"""`), dict literals |
| `{ }` (literal) | Dict literals — **entries separated by `,`** |
| `( )` | Parameter lists, inheritance, call arguments, query lambdas — **arguments separated by `,`** |
| `[ ]` | Array literals, generic type arguments, and **postfix indexing** — elements separated by `,` |
| `;` | Ends attribute declarations and statements (never a separator inside brackets) |
| `,` | Separates arguments, array elements, dict entries, and generic type arguments inside any bracket |
| `,` | Separates array elements, dict entries, and generic type arguments inside `[ ]` and `{ }` |
| `:` | Type annotations, dict key/value separation |
| `::` | Separates the init / condition / update sections of a C-style `for` header |
| `=` | Assignments, default values, named arguments |
| `->` | Return types (methods, functions, query lambdas) |
| `@` | Method and computed property declarations (parameter list `()` mandatory) |
| `.` | Member access and method calls |

---

## 9. Open points (to be decided)

1. **`'...'` strings**: reserved, currently a syntax error (double quotes only,
   by design).
2. **Iterating struct types** (`for(h: Human in Human)`) — supported:
   struct symbols expose `where`, `order_by`, and aggregation helpers
   (§6.7); `for(h: Human in Human.where(...))` iterates the result.
3. **`dict.items()`** — deferred until a pair/tuple type exists.

---

## 10. Parser implementation notes

- Comments **must** be removed before bracket scanning
  (`CommentsRemover.remove_all_comments`); the removal preserves character indices
  (comments become spaces) so positions stay stable.
- `BracketsAndStrings.process` returns top-level `{}` trees (one per class/fn block)
  plus the string span registry; braces inside strings are ignored.
- `NaStructParser.parse_file` dispatches each top-level tree to `parse_struct` /
  `parse_fn`, passing the full header text and the body index range.
- Argument/parameter separation is **depth-aware**: `,` inside nested calls,
  arrays, dicts, or strings never terminates an outer construct
  (`BracketsAndStrings.find_unnested`).
