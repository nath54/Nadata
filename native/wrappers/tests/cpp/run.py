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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "native"))

# pylint: disable=wrong-import-position
import build as dev_build

HERE = Path(__file__).resolve().parent
WRAPPER_DIR = REPO_ROOT / "native" / "wrappers" / "cpp"
IS_WINDOWS = os.name == "nt"
BUILD_DIR = REPO_ROOT / "native" / "build" / (
    "win32" if IS_WINDOWS else sys.platform)

SOURCES = {
    "conformance": HERE / "test_conformance.cpp",
    "bench": HERE / "benchmark.cpp",
}
EXES = {
    "conformance": "test_conformance_cpp.exe"
    if IS_WINDOWS else "test_conformance_cpp",
    "bench": "bench_cpp.exe" if IS_WINDOWS else "bench_cpp",
}


def build(source: Path, kind: str) -> Path:
    """Compiles one test binary against the dev shim import lib."""

    exe = BUILD_DIR / EXES[kind]

    if IS_WINDOWS:
        lib = BUILD_DIR / "nadata.lib"

        if not lib.exists():
            raise SystemExit("dev shim missing — run: python native/build.py")

        command = [
            "cl", "/nologo", "/W4", "/std:c++17", "/EHsc", "/O2",
            f"/I{WRAPPER_DIR}", f"/I{REPO_ROOT / 'native'}",
            str(source),
            "/link", f"/OUT:{exe}",
            f"/LIBPATH:{BUILD_DIR}", "nadata.lib",
        ]
    else:
        if not (BUILD_DIR / "libnadata.so").exists() \
                and not (BUILD_DIR / "libnadata.dylib").exists():
            raise SystemExit("dev shim missing — run: python native/build.py")

        command = [
            os.environ.get("CXX", "c++"), "-std=c++17", "-Wall", "-O2",
            f"-I{WRAPPER_DIR}", f"-I{REPO_ROOT / 'native'}",
            str(source),
            f"-L{BUILD_DIR}", "-lnadata",
            f"-Wl,-rpath,{BUILD_DIR}",
            "-o", str(exe),
        ]

    if dev_build.run_command(command, cwd=BUILD_DIR) != 0:
        raise SystemExit(f"failed to build {EXES[kind]}")

    return exe


def run(exe: Path) -> None:
    """Runs a test binary with the dev shim + runtime reachable."""

    environment = os.environ.copy()

    if "NADATA_HOME" not in environment:
        environment["NADATA_HOME"] = str(REPO_ROOT)

    if IS_WINDOWS:
        base = Path(sys.base_prefix)
        environment["PATH"] = (
            f"{BUILD_DIR};{base};{base / 'DLLs'};"
            + environment.get("PATH", ""))
    else:
        # The dev shim links the system python (rpath already set);
        # the exe finds libnadata via its own rpath.
        environment.setdefault("NADATA_HOME", str(REPO_ROOT))

    result = subprocess.run([str(exe)], env=environment,
                            cwd=str(REPO_ROOT))

    raise SystemExit(result.returncode)


def main() -> None:
    """Parses arguments, builds, runs."""

    parser = argparse.ArgumentParser(
        description="Build & run the C++ wrapper tests")
    parser.add_argument("--bench", action="store_true",
                        help="run the benchmark instead of conformance")
    arguments = parser.parse_args()

    kind = "bench" if arguments.bench else "conformance"
    exe = build(SOURCES[kind], kind)
    run(exe)


if __name__ == "__main__":
    main()
