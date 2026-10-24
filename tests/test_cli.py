"""
Tests for the Nadata command line interface.

Covers the run, info, and dump commands and their error handling.
"""

# Import Modules
import os
from typing import Any

import pytest

from nadata.cli import main


class TestCli:
    """
    Tests the CLI commands end to end.
    """

    def test_run_executes_scripts(self, capsys: Any) -> None:
        """
        Tests that run loads the project and prints script output.
        """

        exit_code: int = main(argv=["run", "./examples/ex0_simple"])
        output: str = capsys.readouterr().out

        assert exit_code == 0
        assert "Adult Parisians count: 3" in output
        assert "Loaded 6 struct(s)" in output

    def test_info_prints_summary(self, capsys: Any) -> None:
        """
        Tests that info prints the database summary.
        """

        exit_code: int = main(argv=["info", "./examples/ex0_simple"])
        output: str = capsys.readouterr().out

        assert exit_code == 0
        assert "Structs (6):" in output
        assert "Human, extends Entity" in output
        assert "Symbols (5): jean, knife, louise, tom, white_cat" in output

    def test_info_on_persisted_file(self, tmp_path: Any, capsys: Any) -> None:
        """
        Tests that info works on a dumped .ndb database file.
        """

        storage_path: str = str(tmp_path / "info_test.ndb")

        assert main(argv=["dump", "./examples/ex0_simple", storage_path]) == 0
        assert main(argv=["info", storage_path]) == 0

        output: str = capsys.readouterr().out

        assert "ObjectInstance: 2 attributes" in output

    def test_dump_persists_file(self, tmp_path: Any) -> None:
        """
        Tests that dump writes a loadable database file.
        """

        storage_path: str = str(tmp_path / "dumped.ndb")

        exit_code: int = main(
            argv=["dump", "./examples/ex0_simple", storage_path]
        )

        assert exit_code == 0
        assert os.path.exists(storage_path)
        assert os.path.getsize(storage_path) > 0

    def test_error_returns_exit_code_one(self, capsys: Any) -> None:
        """
        Tests that runtime errors print to stderr and return 1.
        """

        exit_code: int = main(argv=["run", "./examples/does_not_exist"])

        captured = capsys.readouterr()

        assert exit_code == 1
        assert "error:" in captured.err

    def test_dump_to_existing_file_returns_error(self, tmp_path: Any, capsys: Any) -> None:
        """
        Tests that dumping over an existing file reports an error.
        """

        storage_path: str = str(tmp_path / "exists.ndb")

        storage_file = open(storage_path, "w")
        storage_file.close()

        exit_code: int = main(argv=["dump", "./examples/ex0_simple", storage_path])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "already exists" in captured.err

        os.remove(storage_path)

    def test_missing_command_exits(self) -> None:
        """
        Tests that argparse exits when no command is given.
        """

        with pytest.raises(SystemExit):
            main(argv=[])


class TestCompactCommand:
    """
    Tests the 'nadata compact' command.
    """

    def test_compact_in_place(self, tmp_path: Any, capsys: Any) -> None:
        """
        Tests compacting a database file in place via the CLI.
        """

        from nadata.api import Nadata

        storage_path: str = str(tmp_path / "db.ndb")

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        app.dump(storage_path=storage_path)

        # Append a delta so compaction has something to remove
        opened: Nadata = Nadata.open(storage_path=storage_path)
        opened.execute_data(data_text='zoe = Human(name = "Zoe")')
        opened.save_changes(storage_path=storage_path)

        exit_code: int = main(argv=["compact", storage_path])

        assert exit_code == 0
        assert "->" in capsys.readouterr().out

    def test_compact_to_output(self, tmp_path: Any, capsys: Any) -> None:
        """
        Tests compacting to an explicit output path via the CLI.
        """

        from nadata.api import Nadata

        storage_path: str = str(tmp_path / "db.ndb")
        output_path: str = str(tmp_path / "compact.ndb")

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        app.dump(storage_path=storage_path)

        exit_code: int = main(argv=["compact", storage_path, output_path])

        assert exit_code == 0

        reopened: Nadata = Nadata.open(storage_path=output_path)

        assert "tom" in reopened.symbol_names

    def test_compact_project_folder_fails(self, capsys: Any) -> None:
        """
        Tests that compacting a project folder prints a helpful error.
        """

        exit_code: int = main(argv=["compact", "./examples/ex0_simple"])

        assert exit_code == 1
        assert "dump it first" in capsys.readouterr().err
