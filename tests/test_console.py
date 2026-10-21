"""
Tests for the interactive console mode.
"""

# Import Modules
from __future__ import annotations
from typing import Any

import pytest

from src.api import Nadata
from src.console import NadataConsole


class TestNadataConsole:
    """
    Tests line execution against a loaded database.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example and opens a console over it.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")
        self.console: NadataConsole = NadataConsole(app=self.app)

    def test_expression_echoes_value(self) -> None:
        """
        Tests that a bare expression returns its formatted value.
        """

        assert self.console.execute_line(line="tom.name") == "Tom"

    def test_expression_entity_id(self) -> None:
        """
        Tests that member expressions evaluate (tom.id echoes the index).
        """

        assert self.console.execute_line(line="tom.id") == "0"

    def test_type_expression(self) -> None:
        """
        Tests that type() expressions echo the type handle.
        """

        assert self.console.execute_line(line="type(tom)") == "<Type Human>"

    def test_print_is_silent(self) -> None:
        """
        Tests that None-returning calls (print) echo nothing.
        """

        assert self.console.execute_line(line='print("hello");') is None

    def test_assignment_persists_across_lines(self, capsys: Any) -> None:
        """
        Tests that script variables survive from line to line.
        """

        self.console.execute_line(line="answer = 42;")
        self.console.execute_line(line="print(answer);")

        assert "42" in capsys.readouterr().out

    def test_declaration_registers_symbol(self, capsys: Any) -> None:
        """
        Tests that instance declarations become queryable symbols.
        """

        self.console.execute_line(line='zoe = Human(name = "Zoe")')

        assert "zoe" in self.app.symbol_names

        self.console.execute_line(line="print(zoe.name);")

        assert "Zoe" in capsys.readouterr().out

    def test_unknown_identifier_raises(self) -> None:
        """
        Tests that unresolvable expressions raise NameError.
        """

        with pytest.raises(NameError):
            self.console.execute_line(line="no_such_symbol.field")

    def test_meta_commands(self) -> None:
        """
        Tests that the console commands return their output.
        """

        help_output: str = self.console.execute_line(line=":help") or ""
        classes_output: str = self.console.execute_line(line=":classes") or ""
        symbols_output: str = self.console.execute_line(line=":symbols") or ""
        info_output: str = self.console.execute_line(line=":info") or ""

        assert "Commands:" in help_output
        assert "Entity" in classes_output
        assert "tom" in symbols_output
        assert "struct" in info_output.lower()


class TestConsoleRun:
    """
    Tests the interactive loop over mocked standard input.
    """

    def test_run_processes_lines_and_exits(self, capsys: Any, monkeypatch: Any) -> None:
        """
        Tests that run() evaluates piped lines and exits on :quit.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        console: NadataConsole = NadataConsole(app=app)

        lines: list[str] = ["tom.name", "", ":quit"]

        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: lines.pop(0) if lines else ":quit",
        )

        exit_code: int = console.run()

        assert exit_code == 0
        assert "Tom" in capsys.readouterr().out

    def test_run_survives_error_lines(self, capsys: Any, monkeypatch: Any) -> None:
        """
        Tests that a failing line prints an error and keeps the loop alive.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        console: NadataConsole = NadataConsole(app=app)

        lines: list[str] = ["no_such_symbol.field", "", "tom.name", "", ":quit"]

        monkeypatch.setattr(
            "builtins.input",
            lambda prompt: lines.pop(0) if lines else ":quit",
        )

        exit_code: int = console.run()

        assert exit_code == 0

        out: str = capsys.readouterr().out

        assert "error:" in out
        assert "Tom" in out
