# Development

Conventions, checks, and workflow for working on the Nadata codebase.

---

## Setup

```bash
pip install -e .[dev]      # nadata + pytest + mypy
```

## Checks before every commit

```bash
python -m pytest tests/ -q        # the full test suite must pass
python -m mypy nadata/               # strict type check must stay clean
```

Both gates run quickly (the suite is unit-fast; nothing spawns processes).
The same gates run automatically on every push and pull request via GitHub
Actions (`.github/workflows/ci.yml`), on Linux and Windows across Python
3.10 and 3.13, together with an end-to-end smoke job that exercises the
installed console script against every example project. Run the smoke test
locally with `python scripts/smoke.py`.

## Project layout

```text
nadata/                           # the Python package
    api.py                        # Nadata facade (public Python API)
    cli.py                        # argparse CLI (`nadata` console script)
    console.py                    # interactive console
    session.py                    # session helpers
    __main__.py                   # python -m nadata
    core/
        database/                 # NaDataDatabase, instances, querysets
        database/persistence/     # .ndb format: codec, record store, dump/load
        database/slotted_page.py  # page storage
        expressions/              # ExecutionScope, AST nodes, statements
        struct/                   # Struct, StructAttribute, StructMethod
        types/                    # type system + home types (DateTime, Color)
        values/                   # typed Value wrappers
    parser/
        expressions/              # tokenizer, splitters, expression/statement/type parsers
        nastruct_parser/          # .nastruct IR + resolution phase
        nadata_loader/            # .nadata loader (three-pass instantiation)
examples/                         # 14 runnable example projects (ex0..ex13)
tests/                            # pytest suite (1000+ cases)
docs/                             # documentation (this folder)
native/                           # C ABI shim + language wrappers
    nadata.h                      # the ABI header (generated contract)
    build.py                      # dev shim build + smoke test
    cmake/nadataConfig.cmake      # consumer CMake package
    src/                          # shim implementation (per-domain files)
    wrappers/
        cpp/nadata.hpp            # header-only C++17 wrapper
        java/                     # JNA binding + self-extracting loader
        kotlin/                   # idiomatic Kotlin layer
        tests/                    # conformance + benchmarks per language
scripts/
    build_native.py               # staged dist per OS (release artifact)
    build_jvm_bundle.py           # fat jar packaging
    install.sh / install.ps1      # one-line consumer setup
    release.py, smoke.py, profile.py
benchmarks/                       # engine vs C++ vs JVM vs SQLite
release/                          # release notes kit
.github/workflows/
    ci.yml                        # pytest + mypy + smoke (py3.10-3.13)
    native.yml                    # 4-OS shim build + conformance + [dist] artifacts
```

## Code style

The style rules are enforced by review; `mypy --strict` and the 100-character
line limit are the hard gates.

- **Full type hints** for every variable declaration, every function argument
  and output. Use builtin generics (`list[...]`, `dict[...]`, `tuple[...]`,
  `set[...]`), `X | Y` unions, and `Optional[...]` instead of `| None`. Use
  `typing.Any` rather than bare `any`/`object`.
- **Google-style docstrings with types** on every function, method, and
  class:

  ```python
  def resolve(self) -> NaDataDatabase:
      """
      Resolves every tmp struct and function into the database.

      Returns:
          NaDataDatabase: The populated database.

      Raises:
          TypeError: On unknown structs or inheritance cycles.
      """
  ```

- **Import blocks**: a `# Import Modules` comment before the imports, then
  imports grouped and sorted by module name length.
- **No nested functions** in `src/` — extract closures into small classes or
  module-level helpers (see `LambdaPredicate` for the pattern).
- **100-character limit** per line.
- Blank-line conventions: two blank lines between top-level definitions, one
  blank line between logical blocks inside a function body.

## Commits

Conventional commits with detailed bodies:

```text
feat: incremental persistence and raw source execution in the API

Incremental persistence (log structured replay):

- RecordStore gains a next_document_id base ...
```

- `feat:` — new capability
- `fix:` — bug fix
- `refactor:` — restructuring without behaviour change
- `doc:` — documentation only

Stage logically related changes together; run pytest and mypy before every
commit.

## Testing

The suite lives in `tests/` and runs against the bundled examples, which are
treated as acceptance material:

| File | Covers |
|---|---|
| `test_parser.py` | `.nastruct` scan phase |
| `test_expression_parser.py` | Tokenizer, expression/statement/type parsers |
| `test_expressions.py` | AST evaluation semantics |
| `test_core.py` | Core objects: structs, types, values |
| `test_resolution.py` | Resolution phase, explicit self, shadowing rules |
| `test_nadata_loader.py` | `.nadata` loading, scripts, queries |
| `test_database.py` | Tables, entity ids, querysets |
| `test_persistence.py` | Dump, load, incremental appends |
| `test_api.py` | The `Nadata` facade |
| `test_cli.py` | The command line interface |
| `test_abi_conformance.py` | The canonical ABI case registry (bindings replay it) |
| `test_abi_header_sync.py` | nadata.h stays in sync with the shim |
| `test_queryset.py` | QuerySet chaining, filters, aggregations |
| `test_nid.py` | Stable `nid` identity across loads |
| `test_session.py` | Session semantics |
| `test_persistence.py` / `test_binary_persistence.py` | Dumps, incremental appends, codec |
| `test_memory.py` | Cache eviction, pin/unpin |
| `test_autograd.py` + tensor suites | Tensors, gradients, training builtins |
| `test_spatial_index.py` + geo suites | Geospatial layer |

When adding a language feature, add a case to the relevant example project
*and* a test — the examples double as living documentation.

## Debugging the parser

Inspect the intermediate representation before resolution:

```bash
python -m scripts.inspect_parse examples/ex0_simple
python -m scripts.inspect_parse examples/ex0_simple --json
```

## Native ABI & wrappers

The C ABI shim and the language wrappers are built and tested
separately from the Python suite:

```bash
python native/build.py --run                      # dev shim + 47-check smoke
python native/wrappers/tests/cpp/run.py           # C++ conformance (78 cases)
python native/wrappers/tests/cpp/run.py --bench   # C++ benchmark
python native/wrappers/tests/java/run.py          # Java conformance (72 cases)
python native/wrappers/tests/kotlin/run.py        # Kotlin conformance (76 cases)
python scripts/build_native.py --run              # staged dist + validation
python benchmarks/run_all.py                      # cross-impl benchmarks
```

`tests/abi/contract.py` defines the status-code contract and
`tests/abi/fixtures.py` the shared fixtures; `tests/abi_conformance.py`
(the `@abi_case` registry) is the canonical case list every wrapper
replays. Commit prefix `[dist]` triggers the release-artifact CI job
(`.github/workflows/native.yml`).

## Never run long processes unguarded

When running examples or tests manually, always bound the run (e.g.
`timeout 60 python nadata.py run examples/ex0_simple`). A parser or loader
bug can otherwise loop forever.
