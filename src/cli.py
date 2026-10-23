"""
Command line interface for Nadata.

Provides the run, info, and dump commands over the Nadata high level API.
"""

# Import Modules
from __future__ import annotations
import argparse
import os
import sys
from typing import Any, Optional

from src.api import Nadata
from src.console import NadataConsole
from src.core.database.persistence import compact_database
from src.session import load_folder_session


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
    run_parser.add_argument(
        "script",
        nargs="?",
        default=None,
        help="Optional #manual script to run after the session pipeline.",
    )
    run_parser.add_argument(
        "param",
        nargs="*",
        default=[],
        help="Script parameters as name=value pairs.",
    )
    run_parser.add_argument("--db", default=None, help="World database file.")
    run_parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Skip #auto scripts.",
    )
    run_parser.add_argument(
        "--force-init",
        action="store_true",
        help="Re-run #init scripts even if their markers exist.",
    )

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

    open_parser: argparse.ArgumentParser = sub_parsers.add_parser(
        "open",
        help="Open a project folder session and drop into the console.",
    )
    open_parser.add_argument("folder", help="Project folder.")
    open_parser.add_argument(
        "--db",
        default=None,
        help="World database file (defaults to <folder>/nadata.ndb).",
    )
    open_parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Skip #auto scripts (deterministic opens for tests/CI).",
    )
    open_parser.add_argument(
        "--force-init",
        action="store_true",
        help="Re-run #init scripts even if their markers exist.",
    )

    check_parser: argparse.ArgumentParser = sub_parsers.add_parser(
        "check",
        help="Run a folder session pipeline without a console (CI gate).",
    )
    check_parser.add_argument("folder", help="Project folder.")
    check_parser.add_argument("--db", default=None, help="World database file.")
    check_parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Skip #auto scripts.",
    )
    check_parser.add_argument(
        "--force-init",
        action="store_true",
        help="Re-run #init scripts even if their markers exist.",
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
            return _run_command(
                path=args.path,
                script=args.script,
                params=_parse_param_pairs(pairs=args.param),
                db_path=args.db,
                run_auto=not args.no_auto,
                force_init=args.force_init,
            )

        if args.command == "open":
            return _open_command(
                folder=args.folder,
                db_path=args.db,
                run_auto=not args.no_auto,
                force_init=args.force_init,
            )

        if args.command == "check":
            return _check_command(
                folder=args.folder,
                db_path=args.db,
                run_auto=not args.no_auto,
                force_init=args.force_init,
            )

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


def _parse_param_pairs(pairs: list[str]) -> dict[str, str]:
    """
    Parses name=value CLI pairs into a dict.

    Args:
        pairs (list[str]): Raw 'name=value' strings.

    Returns:
        dict[str, str]: Parsed parameters.

    Raises:
        ValueError: On a pair without '='.
    """

    params: dict[str, str] = {}

    for pair in pairs:

        name, sep, value = pair.partition("=")

        if not sep:
            raise ValueError(
                f"Malformed parameter '{pair}' (expected name=value)"
            )

        params[name] = value

    return params


def _print_session_banner(nadata: Nadata) -> None:
    """
    Prints the loaded-session summary line.

    Args:
        nadata (Nadata): The session facade.
    """

    print(
        f"Loaded {len(nadata.struct_names)} struct(s), "
        f"{nadata.total_instance_count()} instance(s), "
        f"{len(nadata.function_names)} function(s)."
    )


def _run_command(
    path: str,
    script: Optional[str] = None,
    params: Optional[dict[str, str]] = None,
    db_path: Optional[str] = None,
    run_auto: bool = True,
    force_init: bool = False,
) -> int:
    """
    Handles the run command: loads the project, executing its scripts.

    With a script argument this is a headless session run: pipeline
    (#init once, #auto) followed by the named #manual script. Without
    one, the legacy project load applies.

    Args:
        path (str): Project folder or schema file.
        script (Optional[str], optional): #manual script to run.
        params (Optional[dict[str, str]], optional): Script parameters.
        db_path (Optional[str], optional): World database file.
        run_auto (bool, optional): Run #auto scripts. Defaults True.
        force_init (bool, optional): Re-run #init. Defaults False.

    Returns:
        int: Exit code.
    """

    if script is None:
        nadata: Nadata = Nadata.load(path=path)

        _print_session_banner(nadata=nadata)

        return 0

    nadata = Nadata.open_folder(
        folder_path=path,
        db_path=db_path,
        run_auto=run_auto,
        force_init=force_init,
    )

    nadata.execute_script_file(file_path=script, params=params or {})

    _print_session_banner(nadata=nadata)

    return 0


def _open_command(
    folder: str,
    db_path: Optional[str],
    run_auto: bool,
    force_init: bool,
) -> int:
    """
    Handles the open command: session pipeline, then the console.

    Args:
        folder (str): Project folder.
        db_path (Optional[str]): World database file.
        run_auto (bool): Run #auto scripts.
        force_init (bool): Re-run #init scripts.

    Returns:
        int: Exit code.
    """

    nadata: Nadata = Nadata.open_folder(
        folder_path=folder,
        db_path=db_path,
        run_auto=run_auto,
        force_init=force_init,
    )

    _print_session_banner(nadata=nadata)

    console: NadataConsole = NadataConsole(app=nadata)

    return console.run()


def _check_command(
    folder: str,
    db_path: Optional[str],
    run_auto: bool,
    force_init: bool,
) -> int:
    """
    Handles the check command: pipeline without a console (CI gate).

    Args:
        folder (str): Project folder.
        db_path (Optional[str]): World database file.
        run_auto (bool): Run #auto scripts.
        force_init (bool): Re-run #init scripts.

    Returns:
        int: 0 when the pipeline is clean, 1 otherwise.
    """

    database: Any
    result: Any

    database, result = load_folder_session(
        folder_path=folder,
        db_path=db_path,
        run_auto=run_auto,
        force_init=force_init,
    )

    print(result.summary())

    return 0 if result.ok() else 1


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
