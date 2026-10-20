"""
Interactive console mode for Nadata.

After loading a project folder or a persisted .ndb database, the console
evaluates one statement batch per line: bare expressions echo their value,
declarations and scripts execute against the live database, and all state
(instances, symbols, and script variables) persists across lines.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.api import Nadata
from src.core.expressions.expression import Expression
from src.core.expressions.statement import ExprStmt, Statement
from src.core.values.values import Value, ValueExternal
from src.parser.nadata_loader.loader import NaDataLoader, format_runtime_value

# Commands handled by the console itself, not the language
CONSOLE_COMMANDS: dict[str, str] = {
    ":help": "Show this help.",
    ":info": "Print a summary of the loaded database.",
    ":classes": "List the registered structs.",
    ":symbols": "List the bound data symbols.",
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
        print("Type an expression, a statement, or :help for commands.")

        while True:
            try:
                line: str = input("nadata> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0

            if not line:
                continue

            if line in (":quit", ":q", "exit", "quit"):
                return 0

            try:
                output: Optional[str] = self.execute_line(line=line)

                if output is not None:
                    print(output)

            except (NameError, AttributeError, TypeError, KeyError, SyntaxError) as error:
                print(f"error: {error}")

    def execute_line(self, line: str) -> Optional[str]:
        """
        Executes a single console line against the live database.

        A line holding exactly one bare expression echoes its formatted
        value; anything else (declarations, assignments, multi statements)
        executes through the loader. print() and other None-returning calls
        stay silent.

        Args:
            line (str): The source line to execute.

        Returns:
            Optional[str]: The echoed value for expression lines, None
            otherwise.
        """

        if line in CONSOLE_COMMANDS:
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

        if command == ":help":
            lines: list[str] = ["Commands:"]

            for name, description in CONSOLE_COMMANDS.items():
                lines.append(f"  {name:<10} {description}")

            lines.append("")
            lines.append("Anything else is evaluated as Nadata source.")

            return "\n".join(lines)

        if command == ":info":
            return self.app.summary()

        if command == ":classes":
            return "\n".join(self.app.struct_names)

        if command == ":symbols":
            return "\n".join(self.app.symbol_names)

        raise ValueError(f"Unknown command '{command}'")

