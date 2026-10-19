"""
Command line interface for Nadata.

Provides the run, info, and dump commands over the Nadata high level API.
"""

# Import Modules
from __future__ import annotations
import argparse
import sys
from typing import Optional

from src.api import Nadata


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

    args = arg_parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run_command(path=args.path)

        if args.command == "info":
            return _info_command(path=args.path)

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
