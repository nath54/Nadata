"""
python_benchmark.py — the Python-facade implementation of the
cross-language benchmark protocol (native/wrappers/tests/README.md).

Runs the SAME six workloads with the SAME sizes as the C++ reference
(native/wrappers/tests/cpp/benchmark.cpp) and prints the same
parse-friendly format. Every workload's output is verified via
`check <name>: ok` lines; the process exits non-zero on any failure.

Usage: python benchmarks/python_benchmark.py [path-to-ex0_simple]
(run from the repository root; the db is modified in memory only)
"""

# Import Modules
import ctypes
import ctypes.wintypes
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nadata.api import Nadata  # noqa: E402

CHECKS_FAILED = False


def check(name: str, ok: bool) -> None:
    """Prints one protocol check line; tracks failures."""

    global CHECKS_FAILED

    print(f"check {name}: {'ok' if ok else 'FAIL'}")

    if not ok:
        CHECKS_FAILED = True


def peak_rss_bytes() -> int:
    """Best-effort peak RSS (Windows + Linux)."""

    if platform.system() == "Windows":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.POINTER(Counters),
            ctypes.wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

        handle = kernel32.GetCurrentProcess()
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE

        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                      counters.cb):
            return counters.PeakWorkingSetSize

        return 0

    if platform.system() == "Linux":
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024

    return 0


def bench_line(name: str, ops: int, start: float) -> None:
    """Prints one protocol bench line for a finished workload."""

    ms = (time.perf_counter() - start) * 1000.0

    print(f"bench {name}: {ops} ops in {ms:.1f} ms"
          f" => {ops * 1000.0 / ms:.0f} ops/sec")


def main() -> int:
    """Runs the protocol; returns a process exit code."""

    global CHECKS_FAILED

    ex0 = sys.argv[1] if len(sys.argv) > 1 else "./examples/ex0_simple"
    start_rss = peak_rss_bytes()

    # 1. load — N full loads.
    ops = 20
    start = time.perf_counter()

    for _ in range(ops):
        Nadata.load(path=ex0)

    bench_line("load", ops, start)

    db: Nadata = Nadata.load(path=ex0)

    # Verify the fixture before timing anything.
    check("fixture/birthplace",
          db.get_attribute(symbol="tom", name="birthplace") == "Paris")

    # 2. get_attribute.
    ops = 10000
    start = time.perf_counter()

    for _ in range(ops):
        db.get_attribute(symbol="tom", name="birthplace")

    bench_line("get_attribute", ops, start)

    # 3. set_attribute (idempotent write).
    ops = 10000
    start = time.perf_counter()

    for _ in range(ops):
        db.set_attribute(symbol="tom", name="birthplace", value="Lyon")

    bench_line("set_attribute", ops, start)

    check("set_attribute/roundtrip",
          db.get_attribute(symbol="tom", name="birthplace") == "Lyon")

    # 4. query where_str + count (full chain per op) — the same engine
    # path nd_query_where_str uses: parse text, evaluate per candidate.
    from nadata.core.expressions.scope import ExecutionScope
    from nadata.parser.expressions.expression_parser import ExpressionParser

    ops = 1000
    sink = 0
    start = time.perf_counter()

    for _ in range(ops):
        query_set = db.query("Human")
        expression = ExpressionParser.parse_text('h.name == "Tom"')
        query_set = query_set.where_expr(candidate_var="h",
                                         expr=expression,
                                         scope=ExecutionScope())
        sink += query_set.count()

    bench_line("query_where_count", ops, start)

    check("query_where_count/expected", sink == ops)   # exactly one Tom

    # 5. create_instance (grows the db).
    db.execute_schema('class Weapon { name: Str = "Sword"; }')

    ops = 1000
    start = time.perf_counter()

    for i in range(ops):
        db.create_instance(struct_name="Weapon",
                           attributes={"name": f"Blade {i}"})

    bench_line("create_instance", ops, start)

    check("create_instance/count", db.count("Weapon") == 1000)

    # 6. call_function.
    ops = 10000
    sink = 0
    start = time.perf_counter()
    tom = db.get(symbol="tom")

    for _ in range(ops):
        if db.call_function("is_adult", tom):
            sink += 1

    bench_line("call_function", ops, start)

    check("call_function/expected", sink == ops)   # tom (23) is adult

    end_rss = peak_rss_bytes()

    print(f"mem: start RSS {start_rss / (1024 * 1024):.1f} MB,"
          f" peak RSS {end_rss / (1024 * 1024):.1f} MB,"
          f" delta {(end_rss - start_rss) / (1024 * 1024):.1f} MB")

    return 1 if CHECKS_FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
