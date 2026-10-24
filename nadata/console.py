"""
Interactive console mode for Nadata.

After loading a project folder or a persisted .ndb database, the console
evaluates one statement batch per request: source lines accumulate until a
blank line (double ENTER), then execute as a whole — bare expressions echo
their value, declarations and scripts (including multi-line loop bodies)
execute against the live database, and all state (instances, symbols, and
script variables) persists across requests.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Optional

from nadata.api import Nadata
from nadata.core.expressions.expression import Expression
from nadata.core.expressions.statement import ExprStmt, Statement
from nadata.core.expressions.scope import LoopLimitError
from nadata.core.database.persistence.dumper import flush_dirty_instances
from nadata.core.values.values import Value, ValueExternal
from nadata.parser.nadata_loader.loader import NaDataLoader, format_runtime_value
from nadata.session import ScriptFile, parse_script_file, scan_folder

# Commands handled by the console itself, not the language
CONSOLE_COMMANDS: dict[str, str] = {
    ":help": "Show this help.",
    ":info": "Print a summary of the loaded database.",
    ":classes": "List the registered structs.",
    ":symbols": "List the bound data symbols.",
    ":run": "Run a #manual script: :run file.nadata [name=value ...].",
    ":files": "List the session's scripts with their directives.",
    ":reinit": "Re-run #init scripts: :reinit [file.nadata].",
    ":auto": "Re-run the #auto scripts now.",
    ":flush": "Write all dirty instances to the world database.",
    ":quit": "Exit the console (also: :q, exit, quit).",
}


class NadataConsole:
    """
    Interactive console bound to a loaded Nadata database.

    Attributes:
        app (Nadata): The loaded application the console operates on.
        loader (NaDataLoader): Persistent loader whose scope carries state
            across lines.
    """

    def __init__(self, app: Nadata) -> None:
        """
        Initializes the NadataConsole instance.

        Args:
            app (Nadata): Loaded application (project folder, schema file,
                or .ndb database).
        """

        self.app: Nadata = app
        self.loader: NaDataLoader = NaDataLoader(database=app.database)

        # Initialize the persistent scope with the current database state
        self.loader.load_file(content="")

    def run(self) -> int:
        """
        Runs the interactive read-eval-print loop.

        Returns:
            int: Process exit code (0 on normal exit).
        """

        instance_count: int = self.app.total_instance_count()

        print(
            f"Nadata console - {len(self.app.struct_names)} struct(s), "
            f"{instance_count} instance(s), {len(self.app.function_names)} function(s)."
        )
        print(
            "Type an expression or statement, then a blank line to run it "
            "(:help for commands)."
        )

        buffer: list[str] = []

        while True:
            try:
                prompt: str = "nadata> " if not buffer else "      ... "
                line: str = input(prompt).strip()

            except KeyboardInterrupt:
                # Cancel the current request, keep the session alive
                if buffer:
                    buffer = []
                    print("(input cancelled)")
                    continue

                print()

                return 0

            except EOFError:
                print()

                return 0

            # Quit commands work immediately, even mid-request
            if line in (":quit", ":q", "exit", "quit"):
                return 0

            # Commands execute immediately on a single ENTER
            if not buffer and line.startswith(":"):
                print(self._execute_command(command=line))

                continue

            # Blank line: run the accumulated request (if any)
            if not line:
                if not buffer:
                    continue

                source: str = "\n".join(buffer)
                buffer = []

                try:
                    output: Optional[str] = self.execute_line(line=source)

                    if output is not None:
                        print(output)

                except (
                    NameError,
                    AttributeError,
                    TypeError,
                    KeyError,
                    SyntaxError,
                    LoopLimitError,
                ) as error:
                    print(f"error: {error}")

                continue

            buffer.append(line)

    def execute_line(self, line: str) -> Optional[str]:
        """
        Executes one request (single or multi line source) against the
        live database.

        A request holding exactly one bare expression echoes its formatted
        value; anything else (declarations, assignments, loop bodies)
        executes through the loader. print() and other None-returning calls
        stay silent.

        Args:
            line (str): The source request (lines joined with newlines).

        Returns:
            Optional[str]: The echoed value for expression requests, None
            otherwise.
        """

        if line.startswith(":"):
            return self._execute_command(command=line)

        statements: list[Statement] = self.loader.parse_statements(content=line)

        if len(statements) == 1 and isinstance(statements[0], ExprStmt):
            return self._evaluate_expression(expression=statements[0].expression)

        self.loader.load_file(content=line)

        return None

    def _evaluate_expression(self, expression: Expression) -> Optional[str]:
        """
        Evaluates a single expression in the persistent scope.

        Args:
            expression (Any): The expression AST node to evaluate.

        Returns:
            Optional[str]: Formatted value, or None when the expression
            returned no payload (e.g. print, setattr).
        """

        result: Value = expression.evaluate(scope=self.loader.scope)

        if isinstance(result, ValueExternal) and result.payload is None:
            return None

        return format_runtime_value(val=result)

    def _execute_command(self, command: str) -> str:
        """
        Handles a console meta command.

        Args:
            command (str): The command as typed.

        Returns:
            str: The command output.

        Raises:
            ValueError: On unknown commands.
        """

        name: str = command.split(" ", 1)[0].strip()

        handler: str = name

        if handler == ":help":
            lines: list[str] = ["Commands:"]

            for cmd, description in CONSOLE_COMMANDS.items():
                lines.append(f"  {cmd:<10} {description}")

            lines.append("")
            lines.append("Anything else is evaluated as Nadata source.")

            return "\n".join(lines)

        if handler == ":info":
            return self.app.summary()

        if handler == ":classes":
            return "\n".join(self.app.struct_names)

        if handler == ":symbols":
            return "\n".join(self.app.symbol_names)

        if handler == ":flush":
            written: int = self.app.database.flush()

            return f"flushed {written} instance(s)."

        if handler == ":run":
            return self._run_script_command(arguments=command)

        if handler == ":files":
            return self._files_command()

        if handler == ":reinit":
            return self._reinit_command(arguments=command)

        if handler == ":auto":
            return self._auto_command()

        raise ValueError(f"Unknown command '{command}'")

    def _resolve_script(self, file_name: str) -> Any:
        """
        Resolves a script path against the session folder.

        Args:
            file_name (str): Script file name or path.

        Returns:
            Any: The resolved ScriptFile.

        Raises:
            ValueError: If there is no folder session or no such file.
        """

        from pathlib import Path

        if self.app.session_folder is None:
            raise ValueError(
                ":run requires a folder session (nadata open <folder>)"
            )

        folder: Path = Path(self.app.session_folder)
        resolved: Path = Path(file_name)

        if not resolved.is_file():
            resolved = folder / file_name

        if not resolved.is_file():
            raise ValueError(f"Script file not found: '{file_name}'")

        return parse_script_file(path=resolved, folder=folder)

    def _run_script_command(self, arguments: str) -> str:
        """
        Handles ':run file.nadata [name=value ...]'.

        Executes into the console's persistent scope so the script's
        variables stay available afterwards (notebook feel).

        Args:
            arguments (str): Everything after ':run'.

        Returns:
            str: A one-line report.

        Raises:
            ValueError: On missing or mismatched parameters.
        """

        import shlex

        parts: list[str] = arguments.split(" ", 1)
        rest: str = parts[1].strip() if len(parts) > 1 else ""

        # posix=False keeps Windows backslashes in paths intact
        tokens: list[str] = [
            token.strip('"') for token in shlex.split(rest, posix=False)
        ] if rest else []

        if not tokens:
            raise ValueError("Usage: :run file.nadata [name=value ...]")

        script: Any = self._resolve_script(file_name=tokens[0])

        if script.kind != "manual":
            raise ValueError(
                f"'{script.rel_path}' is #{script.kind}: only #manual "
                f"scripts run explicitly"
            )

        from nadata.session import build_prelude

        provided: dict[str, str] = {}

        for token in tokens[1:]:

            key: str
            sep: str
            value: str
            key, sep, value = token.partition("=")

            if not sep:
                raise ValueError(
                    f"Malformed parameter '{token}' (expected name=value)"
                )

            provided[key] = value

        self.loader.prelude = build_prelude(
            params=script.params,
            provided=provided,
        )

        try:
            self.loader.load_file(content=script.text)

        finally:
            self.loader.prelude = {}

        return f"ran {script.rel_path}"

    def _files_command(self) -> str:
        """
        Handles ':files': lists the session's scripts and directives.

        Returns:
            str: The listing (or a note when there is no session).
        """

        from pathlib import Path

        if self.app.session_folder is None:
            return "No folder session (open a project folder)."

        folder: Path = Path(self.app.session_folder)
        _, scripts = scan_folder(folder=folder)

        lines: list[str] = []

        for script in scripts:

            signature: str = ""

            if script.kind == "manual" and script.params:
                signature = "(" + ", ".join(
                    param.name + ": " + param.type_name
                    for param in script.params
                ) + ")"

            lines.append(
                f"  #{script.kind:<7} {script.rel_path}{signature}"
            )

        return "\n".join(lines) if lines else "No .nadata files."

    def _reinit_command(self, arguments: str) -> str:
        """
        Handles ':reinit [file.nadata]'.

        Args:
            arguments (str): Everything after ':reinit'.

        Returns:
            str: A one-line report.

        Raises:
            ValueError: When there is no folder session or no match.
        """

        from pathlib import Path

        if self.app.session_folder is None or self.app.session_db is None:
            raise ValueError(
                ":reinit requires a folder session (nadata open <folder>)"
            )

        parts: list[str] = arguments.split(" ", 1)
        target: str = parts[1].strip() if len(parts) > 1 else ""

        folder: Path = Path(self.app.session_folder)
        _, scripts = scan_folder(folder=folder)

        ran: int = 0

        for script in scripts:

            if script.kind != "init":
                continue

            if target and script.rel_path != target:
                continue

            if target:
                markers: Any = self.app.database.settings.setdefault(
                    "init_done", {}
                )
                markers.pop(script.marker_key(), None)

            loader: NaDataLoader = NaDataLoader(
                database=self.app.database
            )
            loader.load_file(content=script.text)

            self.app.database.settings.setdefault("init_done", {})[
                script.marker_key()
            ] = True

            ran += 1

        if target and ran == 0:
            raise ValueError(f"No #init script matches '{target}'")

        flush_dirty_instances(
            database=self.app.database,
            storage_path=self.app.session_db,
        )

        return f"re-ran {ran} init script(s)."

    def _auto_command(self) -> str:
        """
        Handles ':auto': re-runs every #auto script.

        Returns:
            str: A one-line report.

        Raises:
            ValueError: When there is no folder session.
        """

        from pathlib import Path

        if self.app.session_folder is None:
            raise ValueError(
                ":auto requires a folder session (nadata open <folder>)"
            )

        folder: Path = Path(self.app.session_folder)
        _, scripts = scan_folder(folder=folder)

        ran: int = 0

        for script in scripts:

            if script.kind != "auto":
                continue

            loader: NaDataLoader = NaDataLoader(
                database=self.app.database
            )
            loader.load_file(content=script.text)

            ran += 1

        return f"ran {ran} auto script(s)."

