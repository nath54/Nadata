# Benchmarks & Engine Comparison

How to measure the engine, what the numbers currently look like, how the
engine stores data internally, and an honest comparison with SQLite —
including the parts where Nadata wins for reasons a raw benchmark cannot
show.

---

## 1. Tools

### Stage profiler (per example)

```bash
python scripts/profile.py                  # all examples
python scripts/profile.py ex0_simple ex6_loops_collections
```

Times each pipeline stage on the bundled examples: scan/tokenize, parse
schema, resolve schema, parse data, load data, dump to `.ndb` (+ size),
load from `.ndb`. Single-shot timings — good for *relative* comparison,
not microsecond precision.

### Synthetic benchmark (vs SQLite)

```bash
python benchmarks/bench.py                              # 10000 instances
python benchmarks/bench.py --instances 100000
python benchmarks/bench.py --instances 5000 --skip-sqlite
```

Generates a synthetic `Person` table (typed scalars + an array of tags +
a dict of scores), runs identical operations on both engines, and prints
a comparison table. Both sides use the same seeded data. SQLite stores
`tags` and `scores` as JSON text columns — it has no typed collections,
which is itself part of the comparison.

---

## 2. How the engine manages data internally

**In memory, there is no JSON.** Every instance is a `StructInstance`
object holding `attributes: dict[str, Value]`, where each `Value` is a
typed wrapper (`ValueInt`, `ValueFloat32`, `ValueStr`, `ValueArray`,
`ValueDict`, …) around a raw Python payload. Reading `tom.age` is a dict
lookup plus an attribute access — no parsing, no string decoding, and the
type is already known (schema validation happened at write time, not per
read).

**On disk, rows are positional binary** (format v2) framed into 64 KiB
slotted pages with an LRU cache: each instance is packed against its
struct's declared attribute types (little-endian scalars, length-prefixed
strings, entity-id varints for references), so a typical typed row costs
~66 bytes — slightly *less* than an equivalent SQLite row. Schema,
functions, and symbol documents stay as JSON for inspectability. See
`docs/persistence.md`.

**The memory footprint today is Python-object shaped**: each small int
costs ~28 bytes before the wrapper, and every instance is several dict
entries plus wrapper objects. That is the price of the current
implementation strategy — correct and typed first, compact second. It is
exactly what a future **C or Rust core** would change, with no language
changes:

- column-oriented storage: `Int32` attributes become packed 4-byte arrays
  (SQLite-level memory per row, or better),
- the AST interpreter becomes a compiled bytecode loop,
- the slotted-page layer becomes true fixed-record pages instead of JSON.

The schema language and `.ndb` format are designed so this swap does not
change user-visible behavior — that is why we measure now: today's numbers
are the baseline to beat.

### Planet-scale data

Geospatial workloads stress the same machinery at a different scale: a
world of millions of rows (regions, junctions, buildings, rooms) must
be queryable without decoding it all. The levers, measured by the
streaming scenario in `examples/ex13_earth` (`nadata run
examples/ex13_earth scenarios/stream_district.nadata` under a small
page budget):

- **cell indexing**: `spatial_index` touches only the cells covering a
  query — `locate` / `nearest` / `within` decode candidates, not the
  planet (one index serves a whole struct hierarchy via entity-id
  keys),
- **dump-time row clustering**: rows of a spatially indexed struct are
  written in cell order, so streaming a district reads contiguous
  pages,
- **bounded decode cache**: `clear_cache()` evicts clean pages between
  district walks; pinned/dirty rows stay.

Dedicated micro-benchmarks (cells/s, decode/page under streaming) are
future work; the scenario doubles as the qualitative check.

---

## 3. Current numbers (10000 instances, Windows, Python 3.13)

From `python benchmarks/bench.py --instances 10000 --query-repetitions 5`:

| Operation | Nadata | SQLite |
|---|---:|---:|
| insert / create 10000 rows | 294 ms | 22 ms |
| dump to disk | 1709 ms | — |
| file size | 9.81 MB | 764 KB |
| load from disk | 941 ms | 1.5 ms |
| filtered count (city + age, linear scan) | 4.2 ms | 0.74 ms |
| 100k language-loop iterations | 1537 ms | n/a |
| 1000 schema-method calls | 11 ms | n/a |
| delta update (1000 rows) + save | 1256 ms | 17 ms |

> **Superseded by format v2** (binary rows + lazy loading, see
> `docs/persistence.md`). The same 10000-instance run now reports:
>
> | Operation | v1 | **v2** | SQLite |
> |---|---:|---:|---:|
> | dump to disk | 1709 ms | **258 ms** | — |
> | file size | 9.81 MB | **704 KB** | 764 KB |
> | load from disk | 941 ms | **70 ms** (lazy open) | 1.5 ms |
> | aggregate SUM(age) | — | **3.4 ms** | 0.49 ms |
> | delta update (1000 rows) + save | 1256 ms | **508 ms** | 16 ms |
>
> Nadata's file is now *smaller* than SQLite's while storing typed
> collections, references, and schema-owned methods. The insert, query,
> and language rows are engine-bound and unchanged — that is Phase 4
> (native core) territory. The v1 table is kept as the historical
> baseline.

Reading of the gap:

- **File size is now solved at the encoding level** (format v2): rows are
  positional binary against the schema, landing *below* SQLite's row cost
  while keeping typed collections and references. Remaining overhead is
  page slack and the schema/AST documents, which amortize to nothing.
- **Loading is lazy**: open decodes nothing; scans decode as they go.
  The load-time row above now measures the lazy open (O(meta)).
- **Queries remain the closest race** (~6× slower on a linear scan):
  per-row work is a dict lookup + comparison, and the query engine
  hoists loop-invariant sub-expressions (see `QueryOptimizer`).
- **Bulk insert (~13×) and language loops** are engine-bound: per-instance
  object construction and AST tree-walking. Both are exactly the loops a
  columnar C core would flatten (Phase 4).

These numbers should get *better over time, on the same benchmark* — that
is the point of committing this suite now.

---

## 4. Expressivity: what the comparison cannot show

The benchmark compares row plumbing. The real difference shows in what
each engine *expresses natively*. Take a task like:

> "For every adult Parisian, compute their average score, keep the ones
> above 50, and print a formatted report — plus, scores have per-person
> weighting rules."

**Nadata — schema owns the logic, data owns the types:**

```text
class Person {
    name: Str;
    age: Int32;
    city: Str;
    scores: Dict[Str, Int32] = {};

    @average_score() -> Float32 {
        total: Int32 = 0;
        for(v: Int32 in self.scores.values()){ total += v; }
        return total / self.scores.count();
    }

    @is_target() -> Bool {
        return self.age >= 18 and self.city == "Paris"
            and self.average_score() > 50;
    }
}

# data:
targets = Person.where((p: Person) -> p.is_target());
for(p: Person in targets.to_list()){ print(p.name, p.average_score()); }
```

The logic lives **next to the data**: every reader of the schema sees the
invariants, the CLI runs it with no host language, and the same method
works from the console, the API, and persisted `.ndb` files. `tags` and
`scores` are **typed** — an insert of `scores: {"math": "high"}` is a
TypeError, not a corrupt row.

**SQLite — the logic lives somewhere else:**

```python
# application code (the schema cannot hold this)
cur.execute("""
    SELECT name, scores FROM person
    WHERE age >= 18 AND city = 'Paris'
""")
for name, scores_json in cur.fetchall():
    scores = json.loads(scores_json)          # untyped, runtime errors
    avg = sum(scores.values()) / len(scores)  # the method, re-implemented
    if avg > 50:
        print(name, avg)
```

Every consumer (Python service, CLI tool, notebook) re-implements
`average_score`, and nothing stops `{"math": "high"}` from entering the
JSON column.

### Concrete differences

| Capability | Nadata | SQLite |
|---|---|---|
| Behavior on data (methods) | schema-declared, versioned in `.ndb` | host language or SQL functions, outside the data |
| Typed arrays / dicts | schema-checked at every write | JSON text, unchecked |
| Inheritance & polymorphic queries | native (`DatabaseServer` extends `Server`) | manual join tables or JSON |
| Home types (`DateTime`, `Color`) with arithmetic | native, in-language | strings / extensions |
| Object-graph references (instance refs by entity id) | native, preserved across dump/load | foreign keys + manual joins |
| Schema + data + logic in one portable file | `.ndb` replays everything | schema is SQL, logic is elsewhere |
| Ad-hoc SQL reporting | no (deliberately) | yes, decades of tooling |
| Raw row throughput | ~10–20× slower today | excellent |

**Line count** for the "average score of adult Parisians" task: the
Nadata version above is ~15 lines of schema + 2 lines of query, self
-contained in one file the CLI can run. The SQLite version is ~10 lines
of *host* code — but that code exists **per consumer**, and the average
logic would be duplicated for the CLI tool, the notebook, and the service.
That is the maintenance gap Nadata closes.

### Where SQLite simply wins and should stay winning

- Ad-hoc analytical SQL (`GROUP BY`, joins, window functions). Nadata is
  not aiming to be a reporting engine.
- Multi-writer concurrent access. Nadata is single-writer by design.
- Mature backup/replication tooling.

---

## 5. Running the comparison yourself

```bash
# stage profile of the examples
python scripts/profile.py

# small quick run
python benchmarks/bench.py --instances 2000 --query-repetitions 3

# full scale
python benchmarks/bench.py --instances 100000
```

The benchmark is deterministic (fixed seed): numbers vary across machines
and runs, but relative gaps are stable. When the engine gets an optimizer
or a native core, re-run both tools and update the table in §3.
