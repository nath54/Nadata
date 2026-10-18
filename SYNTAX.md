# Nadata Language Syntax Specification

This document specifies the exact syntax of the two Nadata source file formats:

- **`.nastruct`** — schema files: struct (class) definitions and standalone functions.
- **`.nadata`** — data files: instance declarations, named symbols, and script statements.

The reference examples live in `examples/` (`ex0_simple` to `ex5_edge_cases`) and are
the canonical acceptance material for the parsers.

---

## 1. General rules

| Rule | Value |
|---|---|
| Encoding | UTF-8 |
| Case sensitivity | Fully case-sensitive (`Human` ≠ `human`) |
| Whitespace | Spaces, tabs and newlines are insignificant except inside strings and comments |
| Identifiers | `[A-Za-z_][A-Za-z0-9_]*` (letters, digits, underscore; cannot start with a digit) |
| Keywords | `class`, `fn`, `if`, `else`, `return`, `true`, `false`, `and`, `or`, `not` |

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
- **Floats**: decimal notation with a fractional part (`21.5`, `0.90`). No exponent
  notation (`1e6`) yet.
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
```

### 5.3 Home types

| Type | Construction |
|---|---|
| `DateTime` | `DateTime(y=2003, m=4, d=12)`, `DateTime()` (epoch zero), `DateTime.now()` |
| `Color` | `Color(r, g, b)`, `Color(r, g, b, a)` (channels 0–255) |

`DateTime` field keywords: `y`, `m`, `d`, `H`, `M`, `S`, `MS`.

### 5.4 Struct types

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
- Allowed default expressions: literals (`"Sensor_A"`, `21.5`, `-128`, `true`, `""`),
  constructor calls (`Color(0, 0, 0)`, `DateTime()`), function-like calls
  (`DateTime.now()`), and empty collections (`[]`, `{}`).
- Attributes declared in a parent are inherited; redeclaring the same name in a
  child is an error.

### 6.3 Methods

```text
@name(param: Type; param2: Type = default) -> ReturnType {
    ...body...
}
```

- Method declarations are prefixed with `@`.
- Parameters are `name: Type`, separated by **semicolons `;`** (never commas);
  trailing separators allowed.
- Parameters may have default expressions (evaluated **at each call**, so
  `now: DateTime = DateTime.now()` is refreshed per call).
- **Parameter order is free**: parameters with default values may be declared
  before, after, or between required parameters. Positional call binding follows
  the declared order, named binding follows names — so no ordering constraint
  is needed.
- The return type is mandatory: `-> Type`.
- The body uses the same statement language as `.nadata` scripts (see §7.5).

### 6.4 Computed properties

A method with an **empty parameter list** is a computed property:

```text
@is_broken() -> Bool {
    broken: Bool = false;
    if (state <= 0) {
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
fn name(h1: Human; now: DateTime = DateTime.now()) -> DateTime {
    DateTime age1 = h1.age(now);
    return age1;
}
```

- Declared at top level with the `fn` keyword.
- Same parameter / return type rules as methods (see §6.3).
- Registered globally in the database; callable from methods and scripts.

### 6.6 Statements allowed in method / function bodies

```text
Float32 scaled = x * scale_factor;        // typed variable declaration
broken = true;                            // assignment
if (condition) { ... } else { ... }       // conditionals (if / else)
return result;                            // return (mandatory last expression)
```

- Conditions are parenthesized: `if ((flag_a and not flag_b) or flag_c) { ... }`.
- Operators: `+ - * / %`, comparisons `== != < <= > >=`, logic `and or not`.
- Local variable declarations may be typed (`Float32 result = scaled + bias;`).
- Member access with `.`; method calls with `()`; `self` refers to the receiver.

---

## 7. `.nadata` files — data and scripts

A `.nadata` file mixes three kinds of content: **instance declarations**,
**script statements**, and **queries**.

### 7.1 Call arguments: named or positional, never mixed

Every call site (methods, functions, constructors, computed properties) accepts
exactly one calling style:

```text
obj.method(value1; value2)               // positional, declaration order
obj.method(param1=value1; param2=value2) // named
obj.method()                             // zero args: parentheses still mandatory
```

- **Mixing positional and named arguments in the same call is an error.**
- With named arguments, any parameter order is allowed.
- With positional arguments, values bind in declaration order.
- Trailing separators are allowed in argument lists.

### 7.2 Instance declarations (named symbols)

```text
symbol_name = ClassName(
    attr=value;
    attr2=[sym_a, sym_b];
    attr3={ "key": value };
)
```

- `symbol_name` becomes a database symbol; later declarations can reference it.
- **Forward references are allowed**: an instance may reference symbols declared
  later in the file (including cyclic references).
- Arguments follow the common call rules (§7.1), separated by `;`;
  **trailing separators allowed** (both in argument lists and arrays).
- Empty instantiation is valid: `item2 = EdgeCaseItem()` (all defaults apply).
- Single-line form is valid: `item3 = EdgeCaseItem(x=10; y=20)`.
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
- Built-in `print(...)`: `print("Label:"; value);` — accepts multiple arguments
  separated by `;`.
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
- Returns a queryable collection: `.count()`, `.first()`, chained `.where(...)`.
- The lambda parameter shadows struct members of the same name.
- The query engine pre-evaluates loop-invariant sub-expressions once
  (e.g. `now` in `h.age(now)` is computed a single time, not per candidate).

---

## 8. Punctuation summary

| Token | Usage |
|---|---|
| `{ }` | Class / function bodies, block comments (`"""`), dict literals |
| `{ }` (literal) | Dict literals — **entries separated by `,`** |
| `( )` | Parameter lists, inheritance, call arguments, query lambdas — **arguments separated by `;`** |
| `[ ]` | Array literals and generic type arguments — **elements separated by `,`** |
| `;` | Ends attribute declarations and statements; separates parameters/arguments inside `( )` |
| `,` | Separates array elements, dict entries, and generic type arguments inside `[ ]` and `{ }` |
| `:` | Type annotations, dict key/value separation |
| `=` | Assignments, default values, named arguments |
| `->` | Return types (methods, functions, query lambdas) |
| `@` | Method and computed property declarations (parameter list `()` mandatory) |
| `.` | Member access and method calls |

---

## 9. Open points (to be decided)

1. **`elif`**: not present in examples; `else { if ... }` nesting is the current form.
2. **Loops**: no `for` / `while` syntax defined yet.
3. **Indexing**: no `arr[i]` / `dict["k"]` expression syntax yet (dict/array literals
   exist, element access does not).
4. **String concatenation**: no `+` semantics for `Str` defined yet.
5. **`'...'` strings**: reserved, currently a syntax error.
6. **Exponent numeric literals** (`1e6`): not supported yet.
7. **Multiple assignment targets / augmented assignment** (`+=`): not supported yet.

---

## 10. Parser implementation notes

- Comments **must** be removed before bracket scanning
  (`CommentsRemover.remove_all_comments`); the removal preserves character indices
  (comments become spaces) so positions stay stable.
- `BracketsAndStrings.process` returns top-level `{}` trees (one per class/fn block)
  plus the string span registry; braces inside strings are ignored.
- `NaStructParser.parse_file` dispatches each top-level tree to `parse_struct` /
  `parse_fn`, passing the full header text and the body index range.
