"""
Builds and runs the C++ wrapper test binaries against the dev shim.

Prerequisites:
    - The dev shim built first:  python native/build.py
    - Run from the repository root (fixtures read from ./examples).

Usage:
    python native/wrappers/tests/cpp/run.py             # conformance
    python native/wrappers/tests/cpp/run.py --bench     # benchmark
"""

# Import Modules
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "native"))

# pylint: disable=wrong-import-position
import build as dev_build

HERE = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "native" / "build" / "win32"
SOURCES = [HERE / "test_conformance.cpp", HERE / "benchmark.cpp"]


def build(source: Path, exe_name: str) -> Path:
    """Compiles one test binary against the dev shim import lib."""

    exe = OUT_DIR / exe_name

    if not (OUT_DIR / "nadata.lib").exists():
        raise SystemExit("dev shim missing — run: python native/build.py")

    command = [
        "cl", "/nologo", "/W4", "/std:c++17", "/EHsc", "/O2",
        f"/I{REPO_ROOT / 'native' / 'wrappers' / 'cpp'}",
        f"/I{REPO_ROOT / 'native'}",
        str(source),
        "/link", f"/OUT:{exe}",
        f"/LIBPATH:{OUT_DIR}", "nadata.lib",
    ]

    if dev_build.run_command(command, cwd=OUT_DIR) != 0:
        raise SystemExit(f"failed to build {exe_name}")

    return exe


def run(exe: Path) -> None:
    """Runs a test binary with the dev shim + runtime on PATH."""

    environment = os.environ.copy()

    if "NADATA_HOME" not in environment:
        environment["NADATA_HOME"] = str(REPO_ROOT)

    base = Path(sys.base_prefix)
    environment["PATH"] = (
        f"{OUT_DIR};{base};{base / 'DLLs'};" + environment.get("PATH", ""))

    result = subprocess_run([str(exe)], environment)

    raise SystemExit(result)


def subprocess_run(command: list, environment: dict) -> int:
    """Runs a command, returns its exit code."""

    import subprocess

    return subprocess.run(command, env=environment,
                          cwd=str(REPO_ROOT)).returncode


def main() -> None:
    """Parses arguments, builds, runs."""

    parser = argparse.ArgumentParser(
        description="Build & run the C++ wrapper tests")
    parser.add_argument("--bench", action="store_true",
                        help="run the benchmark instead of conformance")
    arguments = parser.parse_args()

    source = SOURCES[1] if arguments.bench else SOURCES[0]
    exe_name = "bench_cpp.exe" if arguments.bench else "test_conformance_cpp.exe"

    exe = build(source, exe_name)
    run(exe)


if __name__ == "__main__":
    main()
