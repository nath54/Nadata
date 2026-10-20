"""
Command line interface for Nadata.

Provides the run, info, and dump commands over the Nadata high level API.
"""

# Import Modules
from __future__ import annotations
import argparse
import os
import sys
from typing import Optional

from src.api import Nadata
from src.console import NadataConsole
from src.core.database.persistence import compact_database


def main(argv: Optional[list[str]] = None) -> int:
    """
    Runs the Nadata command line interface.

    Args:
        argv (Optional[list[str]], optional): Argument list (defaults to
            sys.argv[1:]). Defaults to None.

    Returns:
        int: Process exit code (0 on success, 1 on error).
    """

    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="nadata",
        description="Nadata declarative data framework command line interface.",
    )

    sub_parsers: argparse._SubParsersAction[argparse.ArgumentParser] = (
        arg_parser.add_subparsers(dest="command", required=True)
    )

    run_parser: argparse.ArgumentParser = sub_parsers.add_parser(
        "run",
        help="Load a project (schemas and data) and execute its scripts.",
    )
    run_parser.add_argument("path", help="Project folder or .nastruct schema file.")

    info_parser: argparse.ArgumentParser = sub_parsers.add_parser(
        "info",
        help="Print a summary of a project or persisted database file.",
    )
    info_parser.add_argument("path", help="Project folder, schema file, or .ndb file.")

    dump_parser: argparse.ArgumentParser = sub_parsers.add_parser(
        "dump",
        help="Load a project and persist it into a slotted page database file.",
    )
    dump_parser.add_argument("source", help="Project folder or schema file.")
    dump_parser.add_argument("destination", help="Target .ndb database file.")

    console_parser: argparse.ArgumentParser = sub_parsers.add_parser(
        "console",
        help="Open an interactive console over a loaded project or database.",
    )
    console_parser.add_argument(
        "path",
        help="Project folder, schema file, or .ndb file.",
    )

    compact_parser: argparse.ArgumentParser = sub_parsers.add_parser(
        "compact",
        help="Rewrite a .ndb file into a minimal snapshot of its current state.",
    )
    compact_parser.add_argument("path", help="The .ndb file to compact.")
    compact_parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional output path (defaults to compacting in place).",
    )

    args = arg_parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run_command(path=args.path)

        if args.command == "info":
            return _info_command(path=args.path)

        if args.command == "console":
            return _console_command(path=args.path)

        if args.command == "compact":
            return _compact_command(path=args.path, output=args.output)

        return _dump_command(source=args.source, destination=args.destination)

    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)

        return 1


def _run_command(path: str) -> int:
    """
    Handles the run command: loads the project, executing its scripts.

    Args:
        path (str): Project folder or schema file.

    Returns:
        int: Exit code.
    """

    nadata: Nadata = Nadata.load(path=path)

    print(
        f"Loaded {len(nadata.struct_names)} struct(s), "
        f"{nadata.total_instance_count()} instance(s), "
        f"{len(nadata.function_names)} function(s)."
    )

    return 0


def _info_command(path: str) -> int:
    """
    Handles the info command: prints a database summary.

    Args:
        path (str): Project folder, schema file, or .ndb file.

    Returns:
        int: Exit code.
    """

    nadata: Nadata = Nadata.load(path=path)

    print(nadata.summary())

    return 0


def _console_command(path: str) -> int:
    """
    Handles the console command: loads the source and opens the REPL.

    Args:
        path (str): Project folder, schema file, or .ndb file.

    Returns:
        int: Exit code.
    """

    nadata: Nadata = Nadata.load(path=path)

    console: NadataConsole = NadataConsole(app=nadata)

    return console.run()


def _compact_command(path: str, output: Optional[str]) -> int:
    """
    Handles the compact command: rewrites a database file to its live state.

    Args:
        path (str): The .ndb file to compact.
        output (Optional[str]): Optional output path (in place when None).

    Returns:
        int: Exit code.
    """

    if os.path.isdir(path):
        print(
            "error: cannot compact a project folder: "
            "dump it first with 'nadata dump'",
            file=sys.stderr,
        )

        return 1

    documents_before: int
    documents_after: int
    documents_before, documents_after = compact_database(
        storage_path=path,
        output_path=output,
    )

    print(
        f"Compacted '{path}': {documents_before} -> {documents_after} "
        f"document(s)."
    )

    return 0


def _dump_command(source: str, destination: str) -> int:
    """
    Handles the dump command: loads the source and persists it.

    Args:
        source (str): Project folder or schema file.
        destination (str): Target .ndb database file.

    Returns:
        int: Exit code.
    """

    nadata: Nadata = Nadata.load(path=source)

    nadata.dump(storage_path=destination)

    print(
        f"Persisted {nadata.total_instance_count()} instance(s) "
        f"across {len(nadata.struct_names)} struct(s) to '{destination}'."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
