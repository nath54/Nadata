"""
Stage profiler for the Nadata engine.

Times each processing stage of the pipeline on the bundled example projects
and reports dump sizes, so engine hot spots are visible in one table:

    1. scan / tokenize  (bracket scan of the schema, token stream of the data)
    2. parse schema     (NaStructParser, block scan + AST)
    3. resolve schema   (types, methods, static checks)
    4. parse data       (statement ASTs for the .nadata scripts)
    5. load data        (declare, populate, execute scripts)
    6. dump to .ndb     (serialization + slotted pages) + file size
    7. load from .ndb   (full replay)

Each stage is timed exactly once (the pipeline is not idempotent), so the
numbers are single-shot measurements - good for relative comparison between
examples, not for microsecond precision.

Usage:
    python scripts/profile.py                     # all examples
    python scripts/profile.py ex0_simple ex6_loops_collections
"""

# Import Modules
from __future__ import annotations
import contextlib
import glob
import io
import os
import sys
import tempfile
import time
from typing import Any

from nadata.core.database.persistence.dumper import dump_database
from nadata.core.database.persistence.loader import load_database
from nadata.parser.expressions.tokenizer import Tokenizer
from nadata.parser.nadata_loader.loader import NaDataLoader
from nadata.parser.nastruct_parser.nastruct_parser import NaStructParser
from nadata.parser.utils.brackets import BracketsAndStrings
from nadata.parser.utils.comments_remove import CommentsRemover

ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Column widths for the report table
COLUMN: int = 24


def _read_project(example_name: str) -> tuple[str, str]:
    """
    Reads and concatenates the schema and data texts of an example.

    Args:
        example_name (str): Example folder name under examples/.

    Returns:
        tuple[str, str]: (schema_text, data_text).
    """

    folder: str = os.path.join(ROOT, "examples", example_name)

    schema_text: str = ""
    schema_files: list[str] = sorted(
        glob.glob(os.path.join(folder, "structs", "*.nastruct"))
    )

    for path in schema_files:
        with open(path, encoding="utf-8") as handle:
            schema_text += handle.read() + "\n"

    data_text: str = ""
    data_files: list[str] = sorted(glob.glob(os.path.join(folder, "data", "*.nadata")))

    for path in data_files:
        with open(path, encoding="utf-8") as handle:
            data_text += handle.read() + "\n"

    return schema_text, data_text


def _human_size(size_bytes: float) -> str:
    """
    Formats a byte count for display.

    Args:
        size_bytes (float): Size in bytes.

    Returns:
        str: Human readable size (e.g. '12.3 KB').
    """

    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"

    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _row(label: str, value: str) -> None:
    """
    Prints one aligned report row.

    Args:
        label (str): Stage label.
        value (str): Formatted value.
    """

    print(f"  {label:<{COLUMN}} {value}")


class StageClock:
    """
    Minimal stopwatch measuring one stage between start() and stop().

    Attributes:
        last_ms (float): Duration of the last measured interval in ms.
    """

    def __init__(self) -> None:
        """
        Initializes the StageClock instance.
        """

        self.last_ms: float = 0.0
        self._start: float = 0.0

    def start(self) -> None:
        """
        Starts the stopwatch.
        """

        self._start = time.perf_counter()

    def stop(self) -> float:
        """
        Stops the stopwatch and records the elapsed time.

        Returns:
            float: Elapsed time in milliseconds.
        """

        self.last_ms = (time.perf_counter() - self._start) * 1000.0

        return self.last_ms


def profile_example(example_name: str) -> None:
    """
    Profiles every pipeline stage of one example project.

    Args:
        example_name (str): Example folder name under examples/.
    """

    clock: StageClock = StageClock()

    schema_text: str
    data_text: str
    schema_text, data_text = _read_project(example_name=example_name)

    print(f"\n=== {example_name} ===")
    print(
        f"  {'source':<{COLUMN}} schema {len(schema_text)} chars, "
        f"data {len(data_text)} chars"
    )

    # 1. Scan / tokenize (standalone passes, no parser state involved)
    clock.start()

    schema_scan: tuple[Any, Any] = BracketsAndStrings.process(
        content=CommentsRemover.remove_all_comments(content=schema_text)
    )
    data_tokens: int = len(
        Tokenizer(
            content=CommentsRemover.remove_all_comments(content=data_text)
        ).tokenize()
    )

    _row(
        label="scan / tokenize",
        value=(
            f"{clock.stop():8.2f} ms  "
            f"({len(schema_scan[0])} schema blocks, {data_tokens} data tokens)"
        ),
    )

    # 2. Parse schema
    parser: NaStructParser = NaStructParser()

    clock.start()
    parser.parse_file(content=schema_text)
    _row(label="parse schema", value=f"{clock.stop():8.2f} ms")

    # 3. Resolve schema (the single resolve: its Struct objects are reused by
    # every later stage, so instance / method types stay consistent)
    clock.start()
    database: Any = parser.resolve()
    _row(label="resolve schema", value=f"{clock.stop():8.2f} ms")

    # 4. Parse data (statement ASTs only, nothing executes)
    parse_loader: NaDataLoader = NaDataLoader(database=database)

    clock.start()
    parse_loader.parse_statements(content=data_text)
    _row(label="parse data", value=f"{clock.stop():8.2f} ms")

    # 5. Load data (declare, populate, execute scripts) - first real load
    load_loader: NaDataLoader = NaDataLoader(database=database)

    clock.start()

    # The data scripts print() during execution: keep the table clean
    with contextlib.redirect_stdout(io.StringIO()):
        load_loader.load_file(content=data_text)

    _row(label="load data", value=f"{clock.stop():8.2f} ms")

    # 6. Dump to .ndb
    temp_dir: str = tempfile.mkdtemp()
    storage_path: str = os.path.join(temp_dir, "profile.ndb")

    clock.start()
    dump_database(database=database, storage_path=storage_path)
    dump_ms: float = clock.stop()

    dump_size: int = os.path.getsize(storage_path)
    _row(
        label="dump to .ndb",
        value=f"{dump_ms:8.2f} ms  ({_human_size(size_bytes=dump_size)})",
    )

    # 7. Load from .ndb (full replay)
    clock.start()
    load_database(storage_path=storage_path)
    _row(
        label="load from .ndb",
        value=f"{clock.stop():8.2f} ms  ({_human_size(size_bytes=dump_size)})",
    )

    os.remove(storage_path)
    os.rmdir(temp_dir)


def main(argv: list[str]) -> int:
    """
    Entry point: profiles the requested examples (or all of them).

    Args:
        argv (list[str]): Example folder names; empty means all.

    Returns:
        int: Process exit code.
    """

    examples: list[str] = [
        name
        for name in sorted(os.listdir(os.path.join(ROOT, "examples")))
        if os.path.isdir(os.path.join(ROOT, "examples", name))
        and name.startswith("ex")
    ]

    if argv:
        examples = argv

    print("Nadata engine stage profile")
    print("(single-shot timings per stage; relative comparison, not u-precision)")

    for example_name in examples:
        profile_example(example_name=example_name)

    return 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv[1:]))
