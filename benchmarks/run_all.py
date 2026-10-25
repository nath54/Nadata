"""
run_all.py — runs the cross-language benchmark protocol and writes
comparable results:

  benchmarks/results.json   machine-readable (all metrics + checks)
  benchmarks/results.md     human-readable comparison table

Implementations (same six workloads, same sizes, same output format):
  - C++      native/wrappers/tests/cpp/benchmark.cpp (via run.py --bench)
  - Python   benchmarks/python_benchmark.py (the nadata facade)
  - SQLite   benchmarks/sqlite_benchmark.py (raw-storage reference)

Every implementation self-verifies its outputs (check <name>: ok
lines); run_all fails if any check fails or an implementation exits
non-zero. Results are still written (marked FAILED) either way.

Usage: python benchmarks/run_all.py [--skip-cpp]
(run from the repository root)
"""

# Import Modules
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

LINE_RE = re.compile(
    r"^bench (?P<name>[a-z_]+): (?P<ops>\d+) ops in (?P<ms>[\d.]+) ms"
    r" => (?P<ops_sec>[\d.]+) ops/sec")
MEM_RE = re.compile(
    r"^mem: start RSS (?P<start>[\d.]+) MB, peak RSS (?P<peak>[\d.]+) MB,"
    r" delta (?P<delta>[\d.]+) MB")
CHECK_RE = re.compile(r"^check (?P<name>\S+): (?P<ok>ok|FAIL)$")


def parse_output(label: str, returncode: int, stdout: str,
                 stderr: str) -> dict:
    """Extracts bench/mem/check entries from protocol output."""

    results = {
        "label": label,
        "exit_code": returncode,
        "bench": {},
        "mem": None,
        "checks": {},
    }

    for line in stdout.splitlines():
        line = line.strip()

        bench = LINE_RE.match(line)

        if bench:
            results["bench"][bench["name"]] = {
                "ops": int(bench["ops"]),
                "ms": float(bench["ms"]),
                "ops_per_sec": float(bench["ops_sec"]),
            }

            continue

        mem = MEM_RE.match(line)

        if mem:
            results["mem"] = {key: float(mem[key])
                              for key in ("start", "peak", "delta")}

            continue

        check = CHECK_RE.match(line)

        if check:
            results["checks"][check["name"]] = check["ok"] == "ok"

    if stderr.strip():
        results["stderr_tail"] = stderr.strip()[-500:]

    return results


def run_implementation(label: str, command: list,
                       env: dict = None) -> dict:
    """Runs one implementation; returns its parsed results dict."""

    print(f"--- {label} ---")

    run = subprocess.run(command, cwd=str(REPO), capture_output=True,
                         text=True, env=env)

    if run.returncode != 0:
        print(f"WARNING: {label} exited {run.returncode}")

    if run.stderr.strip():
        print(f"({label} stderr) {run.stderr.strip()[:300]}")

    return parse_output(label, run.returncode, run.stdout, run.stderr)


def cpp_env() -> dict:
    """Environment for the C++ bench exe (shim + runtime reachable)."""

    env = os.environ.copy()
    env["NADATA_HOME"] = str(REPO)

    if sys.platform == "win32":
        build_dir = REPO / "native" / "build" / "win32"
        base_py = Path(sys.base_prefix)
        env["PATH"] = (f"{build_dir};{base_py};{base_py / 'DLLs'};"
                       + env.get("PATH", ""))
    else:
        build_dir = (REPO / "native" / "build" /
                     ("darwin" if sys.platform == "darwin" else "linux"))
        env["PATH"] = f"{build_dir}:" + env.get("PATH", "")

    return env


def cpp_exe() -> Path:
    """Path of the built benchmark binary."""

    if sys.platform == "win32":
        return REPO / "native" / "build" / "win32" / "benchmark.exe"

    return (REPO / "native" / "build" /
            ("darwin" if sys.platform == "darwin" else "linux") /
            "benchmark")


def collect_cpp() -> dict:
    """Builds (via run.py --bench) and parses the C++ benchmark."""

    harness = (REPO / "native" / "wrappers" / "tests" / "cpp" / "run.py")

    build = subprocess.run([PYTHON, str(harness), "--bench"],
                           cwd=str(REPO), capture_output=True, text=True)

    if build.returncode != 0:
        return {
            "label": "C++ (embedded runtime)",
            "exit_code": build.returncode,
            "bench": {}, "mem": None, "checks": {},
            "stderr_tail": build.stdout[-500:] + build.stderr[-500:],
        }

    # run.py --bench already executed the exe and printed its output;
    # parse that instead of running it twice.
    return parse_output("C++ (embedded runtime)", build.returncode,
                        build.stdout, build.stderr)


def write_markdown(all_results: list, env: dict) -> None:
    """Writes the human-readable comparison table."""

    workload_order = ["load", "get_attribute", "set_attribute",
                      "query_where_count", "create_instance",
                      "call_function"]

    lines = [
        "# Cross-language benchmark results",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Platform: {env['platform']}",
        f"- Python (facade + sqlite): {env['python']}",
        "- C++: reference binding over the C shim (embedded runtime)",
        "",
        "ops/sec, higher is better. C++ and Python run the full nadata",
        "engine (typed attributes, expression evaluation, embedded",
        "interpreter); SQLite is raw row I/O with no engine semantics —",
        "see benchmarks/README.md for the fairness caveats.",
        "",
    ]

    for name in workload_order:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Implementation | ops | ms | ops/sec |")
        lines.append("|---|---|---|---|")

        for result in all_results:
            entry = result["bench"].get(name)

            if entry:
                lines.append(
                    f"| {result['label']} | {entry['ops']} "
                    f"| {entry['ms']:.1f} | {entry['ops_per_sec']:.0f} |")
            else:
                lines.append(f"| {result['label']} | — | — | n/a |")

        lines.append("")

    lines.append("### Memory (peak RSS)")
    lines.append("")
    lines.append("| Implementation | peak MB | delta MB |")
    lines.append("|---|---|---|")

    for result in all_results:
        if result["mem"]:
            lines.append(f"| {result['label']} "
                         f"| {result['mem']['peak']:.1f} "
                         f"| {result['mem']['delta']:.1f} |")
        else:
            lines.append(f"| {result['label']} | n/a | n/a |")

    lines.append("")
    lines.append("### Output verification")
    lines.append("")

    for result in all_results:
        failed = [name for name, ok in result["checks"].items() if not ok]
        status = ("all ok" if not failed
                  else f"**FAILED: {', '.join(failed)}**")

        lines.append(f"- {result['label']}: "
                     f"{len(result['checks'])} checks, {status} "
                     f"(exit {result['exit_code']})")

    lines.append("")

    (BENCH_DIR / "results.md").write_text("\n".join(lines),
                                          encoding="utf-8")


def main() -> int:
    """Runs all implementations, writes results, returns exit code."""

    all_results = []

    if "--skip-cpp" not in sys.argv:
        all_results.append(collect_cpp())

    all_results.append(run_implementation(
        "Python (facade)",
        [PYTHON, str(BENCH_DIR / "python_benchmark.py")]))

    all_results.append(run_implementation(
        "SQLite (raw storage reference)",
        [PYTHON, str(BENCH_DIR / "sqlite_benchmark.py")]))

    env = {
        "platform": (f"{platform.system()} {platform.machine()} "
                     f"({platform.release()})"),
        "python": platform.python_version(),
    }

    (BENCH_DIR / "results.json").write_text(
        json.dumps({"env": env, "results": all_results}, indent=2),
        encoding="utf-8")

    write_markdown(all_results, env)

    ok = all(
        result["exit_code"] == 0
        and all(result["checks"].values())
        and result["bench"]
        for result in all_results
    )

    print(f"\nresults written to benchmarks/results.md and results.json "
          f"({'all checks ok' if ok else 'FAILURES PRESENT'})")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
