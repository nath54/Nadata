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
t = m.transpose();                     // 2-D transpose: new [N, M] copy,
                                       // original untouched
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
passed by identifier; results must satisfy the element type. Note
that `map` cannot change the element type and is gradient-free — for
differentiable whole-op activations use the tensor members `relu`,
`sigmoid`, `gelu`, `softmax` (per-row, stable), and `exp`:

```text
y = x.map((v: Float32) -> v * 2.0);   // lambda
y = x.map(relu);                      // named function (fn declared in schema)
a = z.sigmoid();                      // differentiable activations
p = z.softmax();                      // per-row probabilities (1-D/2-D)
g = z.gelu();                        // tanh-approximated GELU
soft = z.map((v: Float32) -> exp(v)); // compose with the math builtins
soft = soft / soft.sum();             // e.g. a manual softmax
```

**Math builtins** — `abs(v)`, `exp(v)`, `sqrt(v)`, `log(v)`,
`sin(v)`, `cos(v)`, and
`pow(base, exponent)` operate on numeric values (and through `map`
on every tensor cell).

**Shape operations** — `concat` joins tensors (axis 0 by default;
axis 1 for 2-D via a second argument), `reshape` reinterprets the
flat row-major payload with integer dimensions (at most one `-1`,
inferred from the cell count), `flatten` collapses to 1-D, and
`permute` reorders the axes of any rank into a fresh copy (with
`transpose` as the 2-D special case). None of these mutate the
receiver, and a result's new dimensions live on the new value —
assign it to a variable declared with the matching shape:

```text
stacked = a.concat(b);                 // (M,K) + (N,K) -> (M+N,K)
wide    = a.concat(b, 1);              // (M,K) + (M,N) -> (M,K+N)
flat    = img.flatten();               // (M,N) -> (M*N)
img2    = flat.reshape(4, -1);         // -1 inferred from the cell count
swapped = q.permute(2, 0, 1);          // (A,B,C) -> (C,A,B), new tensor
kt      = k.transpose();               // (T,D) -> (D,T), new tensor
```

**Indexing helpers** — `gather` collects rows by integer indices
(embedding lookup; array literals or Int tensors both work), and
`topk(k)` returns the Int64 indices of the k largest cells sorted
descending — compose them for sampling:

```text
looked = table.gather([0, 2, 1]);
top: Tensor[Int64, 3] = logits.topk(3);
best  = logits.gather(logits.topk(1));
```

**Range and randomness** — `arange` builds a 1-D Int64 tensor over
`arange(stop)`, `arange(start, stop)`, or `arange(start, stop, step)`
(positive step, non-empty). `randn()` is a tensor initializer like
`zeros()`: it fills a float tensor with standard-normal samples;
pass an integer seed (`randn(42)`) for deterministic results:

```text
pos: Tensor[Float32, 8] = arange(8).to_float();
w:   Tensor[Float32, 64, 32] = randn(42);
pe:  Tensor[Float32, 8] = pos.map((v: Float32) -> sin(v / 4.0));
```

**Matrix products and statistics** — `matmul` supports all four
rank combinations (dot, matrix-vector, vector-matrix, matrix-matrix);
`argmax`/`argmin` return Int64 indices (per-row on 2-D input); `mean`
returns the average and requires a float element type:

```text
c = a.matmul(b);                      // (M,K) x (K,N) -> (M,N)
d = a.matmul(v);                      // (M,K) x (K) -> (M)
i = scores.argmax();                  // Int64 index (per-row on 2-D)
m = x.mean();                         // average of all cells
```

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

**Autograd** — every Float tensor carries gradient machinery.
Forward operations on tracked tensors record on a per-pass tape;
`backward(loss)` walks it in reverse and deposits gradients onto
**learnable** inputs. `p.w.grad()` returns the accumulated gradient
(a zero tensor when nothing was computed). Only Float tensors
participate; `map` is permanently gradient-free — use explicit
activation members (`relu`, `sigmoid`, `gelu`, `softmax`, `exp`) so
the tape sees whole ops:

```text
loss = (pred - y).mean();              // differentiable reductions
backward(loss);                        // deposit gradients
opt = SGD(lr = 0.05);                  // built-in optimizers
opt.step(p);                           // apply + clear accumulated grads
opt.zero_grad(p);                      // clear without stepping
opt = Adam(lr = 0.001);                // bias-corrected moments
opt.save_to_file("run.opt");           // checkpoint moments + hyperparams
opt.load_from_file("run.opt");         // resume a previous run
```

- Supported ops: arithmetic (`+ - * /`, tensor and scalar chains),
  `matmul` (all four rank cases), `sum`/`mean`/`max`/`min`,
  `transpose`/`permute`/`reshape`/`flatten`, `concat`, `gather`,
  sliced reads, `relu`, `sigmoid`, `gelu`, `softmax` (per-row,
  numerically stable), `exp`, and the pooling/convolution members
  (`max_pool`/`min_pool` route gradients to window winners;
  `avg_pool` spreads them uniformly; `conv1d`/`conv2d` scatter
  transposed products).
- Mutating a tensor between its use and `backward()` is detected via
  version counters and raises a `RuntimeError` — copy or `detach()`
  first.
- `set_learnable(x, false)` stops gradient deposits (recording
  follows); `set_trackable(x, false)` stops op recording without
  touching learnability — both recurse over structs/arrays/tensors.
  Initializers take the same flags: `randn(7, learnable = false)`,
  `zeros(trackable = false)`.
- `array_to_tensor(nested)` converts a rectangular nested array into a
  tensor (shape from the nesting, element type from the declared
  target; ragged nesting is a `ValueError`), and `tensor_to_array(t)`
  round-trips back. `detach(x)` returns an untracked copy.

**Training data** — partitions split a dataset into train/valid/test
with one seed-fixed permutation shared across every partition
(stable membership), largest-remainder sizing, and short tail
batches; an over-run iteration hard-stops with an empty array (reset
the epoch by looping again from iteration 0):

```text
total = get_partition_iteration_count(rows, 7, [0.8, 0.2], 32, 0);
for (it: Int64 = 0 :: it < total :: it++) {
  batch: Array[Array[Float32]] =
    get_batch_of_partition(rows, 7, [0.8, 0.2], 32, 0, it);
  ...
}
```

Weights must be positive and sum to 1 (within 1e-9); each partition
keeps its own iteration counter via the `iteration` argument.

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

### 5.10 Geospatial types

Coordinates live on a planet (sphere or spheroid); `(lat, lon)` are
**geodetic** and the third component is the **radius from the planet's
center** (not an altitude):

```text
Coordinate(43.6, 1.44, 6371008.8)     // geodetic lat, lon, radius (m)
Polyline([Coordinate(...), ...])      // ordered vertex path
Polygon([Coordinate(...), ...])       // closed ring (auto-closing)
```

**Planet** — resolved from per-world settings; scripts set it in `#init`:

```text
set_setting("planet.radius", 6371008.8);   // sphere
set_setting("planet.kind", "spheroid");    // or set equatorial/polar radii
```

**Geodesy builtins** (free functions; `distance(a, b)` and `a.distance(b)`
both work):

| Call | Result |
|---|---|
| `distance(a, b)` | Geodesic distance in meters |
| `bearing(a, b)` / `destination(a, bearing, dist)` | Initial bearing (deg) / destination `Coordinate` |
| `interpolate(a, b, f)` | Point at fraction `f` along the geodesic |
| `area(polygon)` / `perimeter(polygon)` | Spherical area / length in meters |
| `contains(polygon, point)` | Ring containment predicate |
| `circle(center, radius, n)` / `ellipse(...)` / `regular_polygon(...)` | Shape sugar constructors |
| `surface_radius(lat, lon)` | Local planet radius under the point |
| `altitude(coord)` | Radius minus the registered sea surface (see below) |
| `geocentric_latitude(coord)` | Geocentric (vs geodetic) latitude |

**GeoGrid field layers** — a sparse lat/lon grid with the cell size in the
type (arc-seconds; longitude wraps exactly, poles clamp):

```text
GeoGrid[Float64, 3600] elevation = {};   // 1-degree cells, T element
GeoGrid[Str] terrain = {};               // default 30 arc-second cells

elevation.set_at(Coordinate(44.0, 2.0, 6371008.8), 6371108.8);
elevation.at(coord);                     // read a cell (missing raises)
elevation.has(coord);                    // occupancy probe
elevation.sample(coord);                 // bilinear over cell centers;
                                         // falls back to the grid default
elevation.set_default(6371008.8);        // value outside written cells
```

- `sample` weights the four surrounding cell centers bilinearly; missing
  cells drop out and the weights renormalize.
- A grid can be registered as the session's **sea-level surface**;
  `altitude()` then returns the height over the interpolated sea surface
  where the grid has data and over the geometric surface elsewhere:

```text
register_sea_level(earth.elevation);
a = altitude(Coordinate(43.5, 1.5, 6371108.8));   // 100.0 over the bump
clear_sea_level();
```

**Spatial indexing and queries** — session-scoped (rebuilt on open,
like hash indexes), declared over one geometry attribute; a struct
hierarchy shares one index (subclass tables included, parent
references resolved by global entity id):

```text
spatial_index(AdminRegion, "boundary", lod = 8, parent = "parent");
here = locate(AdminRegion, Coordinate(43.6, 1.44, 6371008.8));
                                     // deepest region containing the
                                     // point (hierarchical descent)
near = nearest(AdminRegion, coord, 1);   // Array of the k closest
around = within(Place, coord, 8000.0);   // everything within a radius
```

- `lod` is the cell level of detail: cell edge = 360 / 2^lod degrees.
- `parent = "..."` names a self-reference attribute enabling the
  hierarchical `locate` descent; without it the index is flat.
- Queries take no attribute argument: a struct with exactly one spatial
  index is disambiguated automatically (several indexes raise).

**Routing networks** — nodes are a struct with a `Coordinate`
attribute, edges reference them; edges carry transport modes and
per-mode speeds, and doors / stairs are edges too, so one network
spans streets and interiors:

```text
network_index(Place, Link);
path = route(Coordinate(...), Coordinate(...), "car");
path = route(a, b, "car", cost = "time");   // seconds via speed[mode]
path.edges;                                  // the edges taken
path.cost;                                   // meters (or seconds)
```

- `mode` filters edges: an edge without a `modes` attribute carries
  every mode.
- `cost` is "distance" (default, meters) or "time" (seconds via the
  edge's `speed[mode]`, falling back to its fastest speed).

---

## 6. `.nastruct` files — schemas

A `.nastruct` file is a sequence of top-level blocks: `class` definitions and
standalone `fn` definitions, separated by whitespace and comments.

### 6.0 Imports

A schema file may import other schema files at the top of its source:

```c
import "../shared/base.nastruct";   // relative to the importing file
import "types.nastruct";

class Dog(Animal) { ... }
```

- Imported files are parsed **before** the importer, in dependency
  order — a schema library shared across projects loads transparently.
- Import paths are relative to the importing file's directory (or
  absolute). Cycles raise a clear error naming the chain; a missing
  file raises `FileNotFoundError` with the importer.
- Within one parse, every file is processed exactly once no matter how
  many files import it.
- Imports are a file-level construct: they must each sit on their own
  line, like session directives.

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
- **Error handling**: `raise "message";` aborts execution with a user
  error; `try { ... } catch (err) { ... }` intercepts it. See §6.9.

### 6.9 Error handling: raise, try / catch

```text
try {
    data = read_json("maybe-missing.json");
    if (data.size == 0) { raise "empty dataset"; }
} catch (err) {
    print("failed:", err);
}
```

- `raise <expr>;` — the expression must evaluate to `Str`; an uncaught
  raise aborts the script with the message.
- `try { ... } catch { ... }` — the catch binding is optional. With a
  binding (`catch (err)`), `err` is a `Str` local holding the error
  message, scoped to the handler.
- Catchable errors: language `raise`s **and** runtime errors surfacing
  from the engine (`TypeError`, `ValueError`, `IndexError`, ...), so
  fallible I/O builtins and indexing can be guarded.
- **Never caught**: `return`, `break`, and `continue` unwind through
  protected blocks to their intended target — control flow is not an
  error.
- The loop fuel guard (`LoopLimitError`, §6.8) is catchable, but
  catching it resets nothing: every further loop iteration still
  consumes fuel, so runaway computation stays impossible.

### 6.8 Loops

All loops are **fuel-guarded**: each loop may run at most `loop_limit`
iterations (default **10,000**), then a `LoopLimitError` aborts execution.
The limit is a database setting — change it with the native
`set_loop_limit(n)` from script code or `app.set_loop_limit(n)` from Python;
the value persists in the `.ndb` meta document.

### 6.8b Memory control

File-backed scripts can manage the decode cache and persistence
explicitly (see `docs/persistence.md` for the memory model):

```text
pin(model);                       // keep an instance resident (never evicted)
unpin(model);                     // release the pin
count = flush();                  // write all dirty instances as delta rows
count = clear_cache();            // evict clean, unpinned instances
remove_from_db(old_row);          // remove + append a tombstone (ids never reused)
```

`flush()` and `clear_cache()` return the number of instances written or
evicted; `remove_from_db` persists the removal immediately.

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

### 7.6.1 Filtered sampling

`where(...)` evaluates the predicate on every row. For large tables where
only a handful of matches are needed, the sampling family draws random
rows **without decoding the whole table**:

```text
Item.sample(400);                                        // random rows
Item.sample_by("kind", "qcm", 400);                      // hash-index fast path
Item.sample_where((it: Item) -> it.difficulty < 3, 400); // probed filter
Item.sample_where((it: Item) -> it.retired == false);    // no count: all matches
```

- `sample(count)` — distinct random instances (subclasses included); only
  the chosen rows decode.
- `sample_by(attribute, value, count)` — reads the `index(Struct, "attr")`
  hash bucket directly when one exists (O(bucket), decoding only the drawn
  rows); probes randomly otherwise.
- `sample_where(predicate, count?)` — probes random rows until `count`
  match; if the filter is too sparse the remaining rows are scanned, so
  the result is always correct. Omitting the count returns every match.
- Results are distinct instances in random order; `random.seed(...)` makes
  draws reproducible.
- Sampling draws are plain arrays: iterate them with for-in or index them;
  they are not `queryable` collections (chain `.sample()` after `.where()`
  on the Python `QuerySet` instead).

```text
// Quiz drawing: 20 written questions tagged 'cap-fr', mixing difficulty
Array[Question] picked = Question.sample_where(
    (q: Question) -> q.kind == "written" && "cap-fr" in q.tags, 20
);
```

### 7.7 Named identity — `nid=`

Instances may carry an optional **named id** for stable, human-meaningful
addressing. Identity is opt-in: instances declared without `nid=` are
anonymous and are addressed by `elt.id` (the universal handle).

```c
// First run: creates the instance and registers "temp_living"
s1 = Sensor(nid = "temp_living", label = "Living room");

// Any later run with the same nid: partial update IN PLACE —
// passed attributes are overwritten, the rest are kept, the
// entity id stays stable. Anonymous declarations always create.
Sensor(nid = "temp_living", label = "Living room, west");

// Dynamic nids are expressions, not just literals:
for (i: Int32 in [1, 2, 3]) {
    Child(nid = "child_" + str(i), name = "Child " + str(i));
}
```

Rules:

- The namespace is **per struct**: `nid = "tom"` on `Child` and on `Dog`
  are independent. The internal key is `Child/tom`.
- Lookups walk the **inheritance hierarchy**: `get_by_nid(AdminRegion,
  "france")` finds an instance registered as `Country/france` when
  `Country` inherits `AdminRegion` (useful for hierarchies whose rows
  live in several leaf tables).
- A nid is **immutable** after creation; re-declaring the same nid always
  upserts the same instance.
- `nid=` participates in the two-pass loader like any named argument.

Lookup builtins (O(1) unless noted):

| Call | Result |
|---|---|
| `get_by_nid(Sensor, "temp_living")` | The instance, or `none`-like absence checked Python-side (`None`) when missing |
| `nids(Sensor)` | Sorted `Array[Str]` of every nid in the struct's namespace |
| `elt.nid` | The instance's nid, or an empty payload when anonymous |
| `first(Human)` / `last(Human)` | First / last instance by entity id order |

`str(x)` converts a scalar to its string form (useful for building nids).

### 7.7b String members

String values expose a typed manipulation library (results are `Str`,
`Bool`, or `Array[Str]`; arguments are strictly checked):

| Member | Result |
|---|---|
| `s.upper()` / `s.lower()` / `s.strip()` | `Str` |
| `s.split(sep)` | `Array[Str]` |
| `sep.join(array)` | `Str` (array must be `Array[Str]`) |
| `s.replace(old, new)` | `Str`, all occurrences |
| `s.contains(sub)` / `s.starts_with(p)` / `s.ends_with(sfx)` | `Bool` |
| `s.length()` | `Int64` character count |

Untyped Python `str` methods are not exposed: unknown members raise.

### 7.7c Hash indexes — `index` / `get_by`

`Struct.where(...)` evaluates a predicate per row (parallel chunked
scan). When a scalar attribute is queried by **equality** over and
over, build a session-scoped hash index instead:

```c
index(User, "email");                    // one-time build (materializes rows)
u = get_by(User, "email", "bob@x.io");  // O(1) bucket lookup
```

- Only scalar attributes (`Int` / `Float` / `Str` / `Bool`) are
  indexable; building materializes lazy rows once (a ~30 ms scan for
  50k rows) and every later lookup is O(1) versus a full scan.
- Maintenance is automatic: rows created afterwards join the index
  (schema defaults included), writes re-index, and removals drop out.
- Indexes are **session scoped** — they live in memory and are rebuilt
  by the next `index(...)` call after a reload. Schema evolution drops
  stale columns before indexing, so removed attributes cannot be
  indexed.
- `get_by(Struct, attribute, value)` falls back to a scan of the
  struct's own table when the attribute is not indexed, so scripts
  stay correct either way. It consults the struct's own table only —
  use `Struct.where(...)` for polymorphic scans over subclasses.

Removal: `remove_from_db(elt)` (or `db.remove_instance`) drops the
instance **and** its nid entries. Re-declaring the nid afterwards creates
a **new** entity id — ids are never reused.

### 7.8 Session directives — `#init`, `#auto`, `#manual`

The **first line** of a `.nadata` file may be a directive that tells the
session loader when the script runs. Without a directive the file is
`#manual` (explicit invocation only). Directives are comments to the
parser, so the files remain ordinary scripts.

| Directive | When it runs | Contract |
|---|---|---|
| `#init` | Once per world database | Seeding. A content-hash marker in the world's settings; re-opening skips it. A failed init is a free transaction — the world file is untouched (and a fresh world is not created) |
| `#auto` | Every open (and `:auto`) | Sync/migration. Nothing persists unless the script calls `flush()` |
| `#manual` | Explicit only (`:run`, `nadata run`) | One-shot jobs. Optional parameter signature on the directive line |

`.nastruct` schema files have no directive: they are **re-merged on every
open** — re-declared structs / attributes / methods / functions replace
their previous definitions in place; new ones are added; existing tables,
type ids, and rows are untouched. Schema evolution is an edit + reopen.
Schema files may also `import` shared libraries (see §6.0); in-folder
files parse once, external imports load on every open.

Parameterized manual scripts:

```c
#manual(lr: Float = 0.05, tag: Str = "baseline")
kid = get_by_nid(Child, "tom");
kid.age = kid.age - lr;
flush();
```

- Scalar types only (`Int`/`Int32`/`Int64`, `Float`/`Float32`/`Float64`,
  `Str`, `Bool`); literal defaults. A parameter without a default is
  required; missing required or unknown names are hard errors before any
  statement runs; values are coerced per declared type or error.
- The directive line doubles as self-documentation — `:files` prints it.

### 7.9 I/O builtins

File and settings access from scripts. Paths resolve against the
**project root** and are **sandboxed to it**: relative paths may not
traverse above the root (`../`), and absolute paths are rejected — a
session script can only touch files inside its own project. Without a
project root (plain library use), paths resolve against the current
working directory with no confinement. File effects are **immediate** —
unlike database writes, they are not gated by `flush()`.

| Call | Shape |
|---|---|
| `read_text(path)` / `write_text(path, s)` | whole-file `Str`; writes overwrite, return bytes written |
| `read_lines(path)` / `append_text(path, line)` | `Array[Str]` (no trailing empty entry) / one appended line (file created) |
| `read_json(path)` / `write_json(path, v)` | objects of scalars, arrays of scalars, scalars; writes overwrite |
| `read_csv(path)` / `write_csv(path, rows)` | `Array[dict[Str, Str]]` — stringly-typed by design; first row is the header, all rows must share its keys |
| `exists(path)` / `file_hash(path)` / `list_dir(path)` | `Bool` / sha256 hex `Str` / sorted `Array[Str]` |
| `get_setting(name)` / `set_setting(name, v)` | per-world scalar settings; persisted by the next `flush()` |

```c
cfg = read_json("config.json");          // {"lr": 0.1, "epochs": 3}
lr = cfg["lr"];                          // 0.1 (Float64)
epochs = cfg["epochs"];                  // 3 (Int64)

// Opt-in freshness: re-import only when the input changed
if (get_setting("data_hash") != file_hash("input.csv")) {
    import_data();
    set_setting("data_hash", file_hash("input.csv"));
    flush();
}
```

JSON homogeneity: an object or array's values must all be scalars of the
same family — `int`/`float` mix as the number family (all-int stays
`Int64`); mixed families or nested structures raise with guidance
(stringify or split the file instead). `read_csv` never infers types
(`"007"` stays `"007"`) — convert at the use site with the `int()` /
`float()` builtins (`value = float(row["value"])`); guard reads of
missing files with `exists()`.

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
