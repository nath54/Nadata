"""
Synthetic performance benchmark: Nadata engine versus SQLite.

Generates a synthetic database of Person instances (typed scalars, an array
of tags, and a dict of scores), then measures the same operations on both
engines:

    - bulk insert / create instances   (+ commit)
    - full dump                        (+ on-disk size)
    - full load                        (+ on-disk size)
    - filtered count query             (linear scan on both sides)
    - aggregate (sum)
    - point updates + delta save / SQL UPDATE
    - runtime micro-benches (language loops, method calls)

The goal is NOT to win against SQLite - a mature C engine will crush a
young pure-Python one on raw row operations. The goal is to quantify the
gap, track it as the engine evolves, and document what Nadata offers that
SQL does not (see docs/benchmarks.md).

Usage:
    python benchmarks/bench.py                          # 10000 instances
    python benchmarks/bench.py --instances 100000
    python benchmarks/bench.py --skip-sqlite            # Nadata only
"""

# Import Modules
from __future__ import annotations
import argparse
import json
import os
import random
import sqlite3
import sys
import tempfile
import time
from typing import Any, Callable

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.values.values import make_value

ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed seed so both engines see identical data
SEED: int = 42

CITIES: list[str] = ["Paris", "London", "Berlin", "Madrid", "Rome"]

# Synthetic schema: typed scalars, a typed array, and a typed dict
PERSON_SCHEMA: str = """
class Person {
    name: Str;
    age: Int32;
    height: Float32;
    city: Str;
    tags: Array[Str] = [];
    scores: Dict[Str, Int32] = {};

    @double(x: Int32) -> Int32 {
        return x * 2;
    }

    @category() -> Str {
        if(self.age < 30){ return "young"; }
        if(self.age < 60){ return "adult"; }
        return "senior";
    }
}
"""


def _timed_ms(function: Callable[[], Any]) -> float:
    """
    Runs a callable once and returns its duration in milliseconds.

    Args:
        function (Callable[[], Any]): Operation to measure.

    Returns:
        float: Duration in milliseconds.
    """

    start: float = time.perf_counter()

    function()

    return (time.perf_counter() - start) * 1000.0


def _avg_ms(function: Callable[[], Any], repetitions: int) -> float:
    """
    Runs a callable repeatedly and returns the average duration in ms.

    Args:
        function (Callable[[], Any]): Operation to measure.
        repetitions (int): Number of runs.

    Returns:
        float: Average duration in milliseconds.
    """

    total: float = 0.0

    for _ in range(repetitions):
        total += _timed_ms(function=function)

    return total / repetitions


def _human_size(size_bytes: float) -> str:
    """
    Formats a byte count for display.

    Args:
        size_bytes (float): Size in bytes.

    Returns:
        str: Human readable size.
    """

    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"

    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _row(operation: str, nadata: str, sqlite_result: str) -> None:
    """
    Prints one aligned comparison row.

    Args:
        operation (str): Operation label.
        nadata (str): Nadata measurement.
        sqlite_result (str): SQLite measurement (or '-').
    """

    print(f"  {operation:<34} {nadata:>16} {sqlite_result:>16}")


def _person_attributes(index: int, rng: random.Random) -> dict[str, Any]:
    """
    Builds the attribute payload of one synthetic person.

    Args:
        index (int): Row index.
        rng (random.Random): Seeded generator.

    Returns:
        dict[str, Any]: Attribute values (plain Python types).
    """

    return {
        "name": f"person_{index}",
        "age": rng.randint(18, 80),
        "height": round(rng.uniform(1.5, 2.0), 3),
        "city": rng.choice(CITIES),
        "tags": ["tag_a", "tag_b"] if index % 2 == 0 else ["tag_c"],
        "scores": {"math": index % 100, "sport": (index * 3) % 100},
    }


def _set_raw_attribute(instance: Any, name: str, raw: Any) -> None:
    """
    Sets an instance attribute from a plain Python value.

    Args:
        instance (Any): StructInstance to update.
        name (str): Attribute name.
        raw (Any): Raw value, validated against the declared type.
    """

    attr_type: Any = instance.struct.get_attribute(name=name).attr_type

    instance.set_attribute(name=name, value=make_value(value_type=attr_type, raw=raw))


def _nadata_attribute_getter(instance: Any, name: str) -> Any:
    """
    Reads a raw attribute payload from a StructInstance.

    Args:
        instance (Any): StructInstance to read.
        name (str): Attribute name.

    Returns:
        Any: Raw payload.
    """

    return instance.attributes[name].get()


def run_nadata_benchmark(instances: int, query_repetitions: int, temp_dir: str) -> dict[str, Any]:
    """
    Runs the Nadata side of the benchmark.

    Args:
        instances (int): Number of synthetic people to create.
        query_repetitions (int): Repetitions for the query measurements.
        temp_dir (str): Directory for the .ndb file.

    Returns:
        dict[str, Any]: Measurements keyed by operation name.
    """

    app: Nadata = Nadata(database=NaDataDatabase())

    schema_ms: float = _timed_ms(lambda: app.execute_schema(schema_text=PERSON_SCHEMA))

    rng: random.Random = random.Random(SEED)

    # Payloads pre-generated outside the timing (the SQLite side does too)
    payloads: list[dict[str, Any]] = [
        _person_attributes(index=index, rng=rng) for index in range(instances)
    ]

    def generate() -> None:
        """
        Creates all synthetic instances through the public facade.
        """

        for payload in payloads:
            app.create_instance(struct_name="Person", attributes=payload)

    create_ms: float = _timed_ms(function=generate)

    storage_path: str = os.path.join(temp_dir, "bench.ndb")

    dump_ms: float = _timed_ms(lambda: app.dump(storage_path=storage_path))
    dump_size: int = os.path.getsize(storage_path)

    load_ms: float = _timed_ms(
        lambda: Nadata.open(storage_path=storage_path)
    )

    # Filtered count: linear scan with a raw payload predicate
    def filtered_count() -> int:
        """
        Counts adults in Paris through the query engine.
        """

        return app.query(struct_name="Person").where(
            predicate=lambda inst: (
                _nadata_attribute_getter(instance=inst, name="city") == "Paris"
                and _nadata_attribute_getter(instance=inst, name="age") > 30
            )
        ).count()

    filtered_count()
    query_ms: float = _avg_ms(function=filtered_count, repetitions=query_repetitions)

    def full_scan() -> int:
        """
        Counts every polymorphic instance (full table scan).
        """

        return app.query(struct_name="Person").count()

    full_scan()
    scan_ms: float = _avg_ms(function=full_scan, repetitions=query_repetitions)

    # Runtime micro-bench: language loop over 100k iterations
    app.set_loop_limit(1000000)

    loop_script: str = (
        "s: Int32 = 0;"
        "for(i: Int32 = 0 :: i < 100000 :: i++){ s += i; }"
        'print("loop:", s);'
    )

    import io
    from contextlib import redirect_stdout

    def loop_bench() -> None:
        """
        Runs a 100k-iteration C-style loop through the data loader.
        """

        with redirect_stdout(io.StringIO()):
            app.execute_data(data_text=loop_script)

    loop_bench()
    loop_ms: float = _avg_ms(function=loop_bench, repetitions=3)

    # Method call micro-bench
    app.execute_data(data_text="bench_person = Person(name = \"b\", age = 40);")

    def call_bench() -> None:
        """
        Calls a schema method one thousand times.
        """

        for _ in range(1000):
            app.call_method("bench_person", "double", 21)

    call_ms: float = _timed_ms(function=call_bench)

    # Delta save: touch every 10th instance, then append the delta
    def touch_and_save() -> None:
        """
        Updates every 10th instance and appends the delta to the file.
        """

        for index in range(0, instances, 10):
            app.database.tables_by_struct_name["Person"].instances

        people: list[Any] = app.query(struct_name="Person").to_list()

        for index, person in enumerate(people):
            if index % 10 == 0:
                _set_raw_attribute(instance=person, name="age", raw=99)

        app.save_changes(storage_path=storage_path)

    delta_ms: float = _timed_ms(function=touch_and_save)
    delta_size: int = os.path.getsize(storage_path)

    return {
        "schema_define_ms": schema_ms,
        "create_instances_ms": create_ms,
        "dump_ms": dump_ms,
        "dump_size": dump_size,
        "load_ms": load_ms,
        "filtered_query_ms": query_ms,
        "full_scan_ms": scan_ms,
        "loop_100k_ms": loop_ms,
        "method_1000_calls_ms": call_ms,
        "delta_save_ms": delta_ms,
        "delta_size": delta_size,
    }


def run_sqlite_benchmark(instances: int, query_repetitions: int, temp_dir: str) -> dict[str, Any]:
    """
    Runs the SQLite side of the benchmark on equivalent data.

    Tags and scores are stored as JSON text columns - SQLite has no typed
    collections, which is part of the comparison.

    Args:
        instances (int): Number of synthetic rows to insert.
        query_repetitions (int): Repetitions for the query measurements.
        temp_dir (str): Directory for the .db file.

    Returns:
        dict[str, Any]: Measurements keyed by operation name.
    """

    storage_path: str = os.path.join(temp_dir, "bench.db")

    connection: sqlite3.Connection = sqlite3.connect(storage_path)

    schema_ms: float = _timed_ms(lambda: connection.execute(
        "CREATE TABLE person ("
        " id INTEGER PRIMARY KEY, name TEXT, age INTEGER, height REAL,"
        " city TEXT, tags TEXT, scores TEXT)"
    ))

    rng: random.Random = random.Random(SEED)

    rows: list[tuple[Any, ...]] = []

    for index in range(instances):
        payload: dict[str, Any] = _person_attributes(index=index, rng=rng)

        rows.append((
            payload["name"],
            payload["age"],
            payload["height"],
            payload["city"],
            json.dumps(payload["tags"]),
            json.dumps(payload["scores"]),
        ))

    def insert_all() -> None:
        """
        Bulk inserts every row and commits.
        """

        connection.executemany(
            "INSERT INTO person (name, age, height, city, tags, scores) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

        connection.commit()

    insert_ms: float = _timed_ms(function=insert_all)

    file_size: int = os.path.getsize(storage_path)

    def load_bench() -> None:
        """
        Opens and closes the database file (SQLite has no load step).
        """

        probe: sqlite3.Connection = sqlite3.connect(storage_path)

        probe.close()

    load_ms: float = _timed_ms(function=load_bench)

    def filtered_count() -> int:
        """
        Counts adults in Paris through SQL.
        """

        cursor: sqlite3.Cursor = connection.execute(
            "SELECT COUNT(*) FROM person WHERE city = 'Paris' AND age > 30"
        )

        return int(cursor.fetchone()[0])

    filtered_count()
    query_ms: float = _avg_ms(function=filtered_count, repetitions=query_repetitions)

    def full_scan() -> int:
        """
        Counts every row through SQL.
        """

        cursor: sqlite3.Cursor = connection.execute("SELECT COUNT(*) FROM person")

        return int(cursor.fetchone()[0])

    full_scan()
    scan_ms: float = _avg_ms(function=full_scan, repetitions=query_repetitions)

    def aggregate() -> int:
        """
        Sums every age through SQL.
        """

        cursor: sqlite3.Cursor = connection.execute("SELECT SUM(age) FROM person")

        return int(cursor.fetchone()[0])

    aggregate()
    sum_ms: float = _avg_ms(function=aggregate, repetitions=query_repetitions)

    def update_tenth() -> None:
        """
        Updates every tenth row and commits.
        """

        connection.execute("UPDATE person SET age = 99 WHERE id % 10 = 0")

        connection.commit()

    update_ms: float = _timed_ms(function=update_tenth)

    connection.close()

    return {
        "schema_define_ms": schema_ms,
        "create_instances_ms": insert_ms,
        "dump_ms": 0.0,
        "dump_size": file_size,
        "load_ms": load_ms,
        "filtered_query_ms": query_ms,
        "full_scan_ms": scan_ms,
        "sum_ms": sum_ms,
        "update_ms": update_ms,
    }


def report(nadata_results: dict[str, Any], sqlite_results: Any) -> None:
    """
    Prints the comparison table.

    Args:
        nadata_results (dict[str, Any]): Nadata measurements.
        sqlite_results (Any): SQLite measurements, or None when skipped.
    """

    print("\n=== Results ===")
    _row(operation="operation", nadata="nadata", sqlite_result="sqlite" if sqlite_results else "-")

    _row(
        operation="schema define",
        nadata=f"{nadata_results['schema_define_ms']:.1f} ms",
        sqlite_result=(
            f"{sqlite_results['schema_define_ms']:.1f} ms"
            if sqlite_results else "-"
        ),
    )
    _row(
        operation=f"insert / create {nadata_results.get('instances', '?')} rows",
        nadata=f"{nadata_results['create_instances_ms']:.0f} ms",
        sqlite_result=(
            f"{sqlite_results['create_instances_ms']:.0f} ms"
            if sqlite_results else "-"
        ),
    )
    _row(
        operation="dump to disk",
        nadata=f"{nadata_results['dump_ms']:.0f} ms",
        sqlite_result="-" if not sqlite_results else "0 (no snapshot)",
    )
    _row(
        operation="file size",
        nadata=_human_size(size_bytes=nadata_results["dump_size"]),
        sqlite_result=(
            _human_size(size_bytes=sqlite_results["dump_size"])
            if sqlite_results else "-"
        ),
    )
    _row(
        operation="load from disk",
        nadata=f"{nadata_results['load_ms']:.0f} ms",
        sqlite_result=(
            f"{sqlite_results['load_ms']:.1f} ms" if sqlite_results else "-"
        ),
    )
    _row(
        operation="filtered count (city+age)",
        nadata=f"{nadata_results['filtered_query_ms']:.1f} ms",
        sqlite_result=(
            f"{sqlite_results['filtered_query_ms']:.2f} ms"
            if sqlite_results else "-"
        ),
    )
    _row(
        operation="full scan count",
        nadata=f"{nadata_results['full_scan_ms']:.1f} ms",
        sqlite_result=(
            f"{sqlite_results['full_scan_ms']:.2f} ms" if sqlite_results else "-"
        ),
    )

    if sqlite_results and "sum_ms" in sqlite_results:
        _row(
            operation="aggregate SUM(age)",
            nadata="(see notes)",
            sqlite_result=f"{sqlite_results['sum_ms']:.2f} ms",
        )

    _row(
        operation="100k loop iterations",
        nadata=f"{nadata_results['loop_100k_ms']:.0f} ms",
        sqlite_result="n/a (not a language)",
    )
    _row(
        operation="1000 method calls",
        nadata=f"{nadata_results['method_1000_calls_ms']:.0f} ms",
        sqlite_result="n/a (not a language)",
    )
    _row(
        operation="delta update + save",
        nadata=f"{nadata_results['delta_save_ms']:.0f} ms",
        sqlite_result=(
            f"{sqlite_results['update_ms']:.0f} ms" if sqlite_results else "-"
        ),
    )
    _row(
        operation="file size after delta",
        nadata=_human_size(size_bytes=nadata_results["delta_size"]),
        sqlite_result="-",
    )


def main(argv: list[str]) -> int:
    """
    Entry point: runs both benchmark sides and prints the comparison.

    Args:
        argv (list[str]): Command line arguments.

    Returns:
        int: Process exit code.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Nadata vs SQLite synthetic benchmark"
    )
    parser.add_argument("--instances", type=int, default=10000)
    parser.add_argument("--query-repetitions", type=int, default=5)
    parser.add_argument("--skip-sqlite", action="store_true")
    args: argparse.Namespace = parser.parse_args(argv)

    random.seed(SEED)

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Generating {args.instances} Person instances...")

        nadata_results: dict[str, Any] = run_nadata_benchmark(
            instances=args.instances,
            query_repetitions=args.query_repetitions,
            temp_dir=temp_dir,
        )
        nadata_results["instances"] = args.instances

        sqlite_results: Any = None

        if not args.skip_sqlite:
            sqlite_results = run_sqlite_benchmark(
                instances=args.instances,
                query_repetitions=args.query_repetitions,
                temp_dir=temp_dir,
            )

        report(nadata_results=nadata_results, sqlite_results=sqlite_results)

    return 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv[1:]))
