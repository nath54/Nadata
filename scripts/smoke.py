"""
End-to-end smoke test for the installed Nadata console script.

Runs every bundled example project through 'nadata run' and exercises a
full persistence round trip (dump, info, compact) on a temporary .ndb
file. Exits non-zero on the first failure.

Usage:
    python scripts/smoke.py
"""

# Import Modules
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Project root (the folder containing examples/ and nadata.py)
ROOT: Path = Path(__file__).resolve().parent.parent

# Bundled example projects, run in order
EXAMPLES: list[str] = [
    "ex0_simple",
    "ex1_primitives",
    "ex2_inheritance",
    "ex3_collections",
    "ex4_expressions",
    "ex5_edge_cases",
]

# Hard cap per subprocess, so a hang can never stall the smoke run
COMMAND_TIMEOUT: int = 60


def resolve_nadata() -> str:
    """
    Locates the nadata console script.

    Prefers the script installed next to the current interpreter (venv or
    CI environment); falls back to relying on the process PATH.

    Returns:
        str: Executable path or bare command name.
    """

    scripts_dir: Path = Path(sys.executable).resolve().parent

    for candidate in ("nadata.exe", "nadata.cmd", "nadata"):
        script_path: Path = scripts_dir / candidate

        if script_path.exists():
            return str(script_path)

    return "nadata"


def run_command(args: list[str]) -> None:
    """
    Runs a console command, failing loudly on error or timeout.

    Args:
        args (list[str]): Command arguments (without the executable).
    """

    print(f"  $ nadata {' '.join(args)}")

    subprocess.run(
        args=[resolve_nadata()] + args,
        cwd=str(ROOT),
        check=True,
        timeout=COMMAND_TIMEOUT,
    )


def run_examples() -> None:
    """
    Runs every bundled example project.
    """

    for name in EXAMPLES:
        print(f"Running example {name}...")
        run_command(args=["run", str(ROOT / "examples" / name)])


def run_persistence_round_trip() -> None:
    """
    Exercises dump, info, compact, and a re-opened info on a temp file.
    """

    print("Running persistence round trip...")

    with tempfile.TemporaryDirectory() as temp_dir:
        database_path: str = os.path.join(temp_dir, "smoke.ndb")

        run_command(
            args=["dump", str(ROOT / "examples" / "ex0_simple"), database_path]
        )
        run_command(args=["info", database_path])
        run_command(args=["compact", database_path])
        run_command(args=["info", database_path])


def main() -> int:
    """
    Runs the full smoke suite.

    Returns:
        int: Process exit code (0 on success).
    """

    run_examples()
    run_persistence_round_trip()

    print("Smoke test passed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
