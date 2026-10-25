"""
sqlite_benchmark.py — SQLite as the "raw database" reference point for
the cross-language benchmark protocol (native/wrappers/tests/README.md).

Builds a SQLite mirror of the ex0 fixture (humans + weapons, an
is_adult(age) UDF) and runs the SAME six workloads with the SAME sizes
as the C++ reference. This is intentionally NOT apples-to-apples with
nadata — SQLite does raw row I/O with zero schema/engine semantics,
while nadata pays for typed attribute access, expression evaluation
and an embedded interpreter. Read the numbers as "the price of the
engine vs the price of the storage layer".

Caveats documented in benchmarks/README.md:
- create_instance on SQLite is a durable INSERT; nadata's is in-memory
  until save_changes().
- call_function maps to a per-row UDF call, the closest equivalent.

Usage: python benchmarks/sqlite_benchmark.py [workdir]
(the db file is created inside a temp directory and deleted)
"""

# Import Modules
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from python_benchmark import check, peak_rss_bytes  # noqa: E402

WORKLOAD_OPS = {
    "load": 20,
    "get_attribute": 10000,
    "set_attribute": 10000,
    "query_where_count": 1000,
    "create_instance": 1000,
    "call_function": 10000,
}


def bench_line(name: str, ops: int, start: float) -> None:
    """Prints one protocol bench line for a finished workload."""

    ms = (time.perf_counter() - start) * 1000.0

    print(f"bench {name}: {ops} ops in {ms:.1f} ms"
          f" => {ops * 1000.0 / ms:.0f} ops/sec")


def build_mirror(path: str) -> sqlite3.Connection:
    """Creates the ex0-equivalent schema and seed data."""

    connection = sqlite3.connect(path)

    connection.executescript("""
        PRAGMA journal_mode = MEMORY;
        PRAGMA locking_mode = EXCLUSIVE;   -- single process: skip the
                                           -- per-statement file locks that
                                           -- dominate SQLite on Windows
        CREATE TABLE human (
            id INTEGER PRIMARY KEY,
            symbol TEXT UNIQUE,
            name TEXT,
            birthplace TEXT,
            age INTEGER
        );
        INSERT INTO human (symbol, name, birthplace, age) VALUES
            ('tom', 'Tom', 'Paris', 23),
            ('jean', 'Jean', 'Paris', 33),
            ('louise', 'Louise', 'Paris', 28);
        CREATE TABLE weapon (
            id INTEGER PRIMARY KEY,
            name TEXT
        );
    """)

    # The engine-function equivalent: a UDF called once per row.
    connection.create_function("is_adult", 1, lambda age: age >= 18)

    return connection


def main() -> int:
    """Runs the protocol; returns a process exit code."""

    workdir = tempfile.mkdtemp(prefix="nadata-sqlite-bench-")
    path = os.path.join(workdir, "mirror.db")

    start_rss = peak_rss_bytes()
    connection = build_mirror(path)

    # 2. get_attribute — point read by symbol.
    # (Workloads run 2-6 first: under locking_mode=EXCLUSIVE the mirror
    # connection would lock out the 'load' workload's fresh
    # connections. 'load' runs last; the parser keys results by name.)
    ops = WORKLOAD_OPS["get_attribute"]
    cursor = connection.cursor()
    read_sql = "SELECT birthplace FROM human WHERE symbol = 'tom'"

    start = time.perf_counter()

    for _ in range(ops):
        cursor.execute(read_sql).fetchone()

    bench_line("get_attribute", ops, start)

    check("fixture/birthplace",
          connection.execute(
              "SELECT birthplace FROM human WHERE symbol = 'tom'"
          ).fetchone()[0] == "Paris")

    # 3. set_attribute — idempotent point write.
    ops = WORKLOAD_OPS["set_attribute"]
    cursor = connection.cursor()
    write_sql = "UPDATE human SET birthplace = 'Lyon' WHERE symbol = 'tom'"

    start = time.perf_counter()

    for _ in range(ops):
        cursor.execute(write_sql)

    bench_line("set_attribute", ops, start)

    check("set_attribute/roundtrip",
          connection.execute(
              "SELECT birthplace FROM human WHERE symbol = 'tom'"
          ).fetchone()[0] == "Lyon")

    # 4. query_where_count — filtered count per op.
    ops = WORKLOAD_OPS["query_where_count"]
    cursor = connection.cursor()
    count_sql = "SELECT COUNT(*) FROM human WHERE name = 'Tom'"

    sink = 0
    start = time.perf_counter()

    for _ in range(ops):
        sink += cursor.execute(count_sql).fetchone()[0]

    bench_line("query_where_count", ops, start)

    check("query_where_count/expected", sink == ops)   # exactly one Tom

    # 5. create_instance — durable insert (nadata's is in-memory).
    ops = WORKLOAD_OPS["create_instance"]
    cursor = connection.cursor()
    insert_sql = "INSERT INTO weapon (name) VALUES (?)"

    start = time.perf_counter()

    for i in range(ops):
        cursor.execute(insert_sql, (f"Blade {i}",))

    bench_line("create_instance", ops, start)

    check("create_instance/count",
          connection.execute("SELECT COUNT(*) FROM weapon").fetchone()[0]
          == ops)

    # 6. call_function — UDF called once per op (engine-function price).
    ops = WORKLOAD_OPS["call_function"]
    cursor = connection.cursor()
    udf_sql = "SELECT is_adult(age) FROM human WHERE symbol = 'tom'"

    sink = 0
    start = time.perf_counter()

    for _ in range(ops):
        if cursor.execute(udf_sql).fetchone()[0]:
            sink += 1

    bench_line("call_function", ops, start)

    check("call_function/expected", sink == ops)   # tom (23) is adult

    connection.close()

    # 1. load — full close + reopen + read everything (the closest
    # equivalent of re-loading a project: schema + all rows). Emitted
    # last (see the note above); the parser keys results by name.
    ops = WORKLOAD_OPS["load"]
    start = time.perf_counter()

    for _ in range(ops):
        fresh = sqlite3.connect(path)
        fresh.execute("SELECT COUNT(*) FROM human").fetchone()
        fresh.close()

    bench_line("load", ops, start)

    end_rss = peak_rss_bytes()

    print(f"mem: start RSS {start_rss / (1024 * 1024):.1f} MB,"
          f" peak RSS {end_rss / (1024 * 1024):.1f} MB,"
          f" delta {(end_rss - start_rss) / (1024 * 1024):.1f} MB")

    connection.close()

    try:
        os.remove(path)
        os.rmdir(workdir)
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
