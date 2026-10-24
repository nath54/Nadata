"""
Tests for the folder session layer: directive classification, the
init-once / auto-every / manual-explicit pipeline, parameterized
#manual scripts, and the console session commands.
"""

# Import Modules
import os
import shutil
import tempfile
from typing import Any

import pytest

from nadata.api import Nadata
from nadata.console import NadataConsole
from nadata.session import (
    folder_is_session,
    load_folder_session,
    parse_script_file,
)


def _write(folder: Any, rel: str, text: str) -> None:
    """
    Writes a file into the project folder (creating parents).

    Args:
        folder (Any): Project folder Path.
        rel (str): Relative posix path.
        text (str): File content.
    """

    target: Any = folder / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


SCHEMA: str = (
    "class Child {\n"
    "    name: Str;\n"
    "    age: Int32 = 0;\n"
    "}\n"
)

SEED: str = (
    "#init\n"
    'tom = Child(nid = "tom", name = "Tom", age = 6);\n'
    'anna = Child(nid = "anna", name = "Anna", age = 7);\n'
    "flush();\n"
)

AUTO: str = (
    "#auto\n"
    'Child(nid = "tom", name = "Thomas");\n'
    "flush();\n"
)

TRAIN: str = (
    '#manual(lr: Float = 0.05, tag: Str = "baseline")\n'
    'kid = get_by_nid(Child, "tom");\n'
    "kid.age = 0;\n"
)


def _project(tmp: str, name: str) -> Any:
    """
    Creates a standard project folder.

    Args:
        tmp (str): Temp directory root.
        name (str): Folder name.

    Returns:
        Any: The folder Path.
    """

    from pathlib import Path

    folder: Any = Path(tmp) / name
    folder.mkdir(parents=True, exist_ok=True)

    _write(folder, "schema.nastruct", SCHEMA)
    _write(folder, "seed.nadata", SEED)
    _write(folder, "sync.nadata", AUTO)

    return folder


class TestDirectiveParsing:
    """
    First-line directive classification.
    """

    def test_kinds(self) -> None:
        """
        init/auto/manual are classified from the first non-blank line.
        """

        from pathlib import Path

        folder: Any = Path(tempfile.mkdtemp())

        _write(folder, "a.nadata", "#init\nx = 1;\n")
        _write(folder, "b.nadata", "\n\n  #auto  \ny = 2;\n")
        _write(folder, "c.nadata", "#manual(lr: Float = 0.1)\nz = 3;\n")
        _write(folder, "d.nadata", "w = 4;\n")

        _, scripts = __import__(
            "nadata.session", fromlist=["scan_folder"]
        ).scan_folder(folder=folder)

        by_name: dict[str, Any] = {
            script.rel_path: script for script in scripts
        }

        assert by_name["a.nadata"].kind == "init"
        assert by_name["b.nadata"].kind == "auto"
        assert by_name["c.nadata"].kind == "manual"
        assert by_name["d.nadata"].kind == "manual"

        shutil.rmtree(folder, ignore_errors=True)

    def test_manual_signature(self) -> None:
        """
        Manual parameters parse with names, types, and defaults.
        """

        from pathlib import Path

        folder: Any = Path(tempfile.mkdtemp())

        _write(folder, "t.nadata", '#manual(lr: Float = 0.05, tag: Str = "x")\n')

        script: Any = parse_script_file(
            path=folder / "t.nadata", folder=folder
        )

        assert [p.name for p in script.params] == ["lr", "tag"]
        assert [p.type_name for p in script.params] == ["Float", "Str"]
        assert script.params[0].default_text == "0.05"

        shutil.rmtree(folder, ignore_errors=True)

    def test_folder_is_session(self) -> None:
        """
        Directives or an existing world trigger session semantics.
        """

        from pathlib import Path

        folder: Any = Path(tempfile.mkdtemp())

        assert not folder_is_session(folder=folder)

        _write(folder, "a.nadata", "#init\n")
        assert folder_is_session(folder=folder)

        shutil.rmtree(folder, ignore_errors=True)


class TestPipeline:
    """
    init-once / auto-every / manual-explicit behavior.
    """

    def test_fresh_open_seeds_once(self, tmp_path: Any) -> None:
        """
        First open creates the world and runs init; second skips it.
        """

        folder: Any = _project(str(tmp_path), "proj")

        database: Any
        result: Any

        database, result = load_folder_session(folder_path=str(folder))

        assert result.created_fresh
        assert result.init_ran == ["seed.nadata"]
        assert result.init_skipped == []
        assert result.ok()
        assert (folder / "nadata.ndb").exists()

        # #auto also runs on the first open (after #init)
        assert result.auto_ran == ["sync.nadata"]

        assert database.get_by_nid(
            struct_name="Child", nid="tom"
        ).attributes["name"].get() == "Thomas"

        # Second open: init skipped, auto re-ran
        database, result = load_folder_session(folder_path=str(folder))

        assert not result.created_fresh
        assert result.init_ran == []
        assert result.init_skipped == ["seed.nadata"]
        assert result.auto_ran == ["sync.nadata"]

        # #auto + nid upsert means NO duplication
        assert result.instance_count == 2

    def test_manual_never_runs_automatically(self, tmp_path: Any) -> None:
        """
        #manual scripts are untouched by the pipeline.
        """

        folder: Any = _project(str(tmp_path), "manual")

        _write(folder, "train.nadata", TRAIN)

        database: Any
        result: Any

        database, result = load_folder_session(folder_path=str(folder))
        database, result = load_folder_session(folder_path=str(folder))

        assert result.auto_ran == ["sync.nadata"]
        assert "train.nadata" not in result.auto_ran

    def test_failed_init_leaves_nothing(self, tmp_path: Any) -> None:
        """
        A broken #init on a fresh folder creates no world file.
        """

        folder: Any = _project(str(tmp_path), "broken")

        _write(folder, "seed.nadata", "#init\nChild(nid = \"x\", oops = 1);\n")

        database: Any
        result: Any

        database, result = load_folder_session(folder_path=str(folder))

        assert len(result.init_errors) == 1
        assert not (folder / "nadata.ndb").exists()

    def test_failed_init_existing_world_not_flushed(
        self, tmp_path: Any
    ) -> None:
        """
        A broken #init on an existing world writes nothing (no marker).
        """

        folder: Any = _project(str(tmp_path), "existing")

        load_folder_session(folder_path=str(folder))

        _write(folder, "later.nadata", "#init\nChild(oops = 2);\n")

        database: Any
        result: Any

        database, result = load_folder_session(folder_path=str(folder))

        assert [path for path, _ in result.init_errors] == ["later.nadata"]

        # Reload: the broken init left no marker, still only 2 children
        reloaded: Nadata = Nadata.open(
            storage_path=str(folder / "nadata.ndb")
        )

        assert reloaded.count("Child") == 2

    def test_no_auto_flag(self, tmp_path: Any) -> None:
        """
        --no-auto makes opens deterministic.
        """

        folder: Any = _project(str(tmp_path), "noauto")

        load_folder_session(folder_path=str(folder))

        database: Any
        result: Any

        database, result = load_folder_session(
            folder_path=str(folder), run_auto=False
        )

        assert result.auto_ran == []

        # The first open already applied the auto upsert (Thomas); with
        # --no-auto this open leaves the value frozen
        assert database.get_by_nid(
            struct_name="Child", nid="tom"
        ).attributes["name"].get() == "Thomas"

    def test_force_init_reruns(self, tmp_path: Any) -> None:
        """
        --force-init re-runs init scripts (upsert keeps ids stable).
        """

        folder: Any = _project(str(tmp_path), "force")

        load_folder_session(folder_path=str(folder))

        database: Any
        result: Any

        database, result = load_folder_session(
            folder_path=str(folder), force_init=True
        )

        assert result.init_ran == ["seed.nadata"]
        assert result.init_skipped == []
        assert result.instance_count == 2

    def test_custom_db_path(self, tmp_path: Any) -> None:
        """
        --db worlds/john.ndb keeps worlds independent.
        """

        folder: Any = _project(str(tmp_path), "worlds")

        load_folder_session(
            folder_path=str(folder), db_path="worlds/john.ndb"
        )

        load_folder_session(
            folder_path=str(folder), db_path="worlds/marie.ndb"
        )

        assert (folder / "worlds" / "john.ndb").exists()
        assert (folder / "worlds" / "marie.ndb").exists()
        assert not (folder / "nadata.ndb").exists()


class TestManualParameters:
    """
    Parameterized #manual execution through the facade.
    """

    def _opened(self, tmp_path: Any) -> tuple[Nadata, Any]:
        """
        Opens a session folder with a parameterized training script.

        Args:
            tmp_path (Any): Pytest tmp dir.

        Returns:
            tuple[Nadata, Any]: The app and folder.
        """

        folder: Any = _project(str(tmp_path), "params")
        _write(folder, "train.nadata", TRAIN)

        app: Nadata = Nadata.open_folder(folder_path=str(folder))

        return app, folder

    def test_defaults_and_coercion(self, tmp_path: Any) -> None:
        """
        Params bind with defaults; provided strings coerce per type.
        """

        app: Nadata
        folder: Any

        app, folder = self._opened(tmp_path)

        # Params are plain script variables; expose them via a symbol row
        app.execute_script_file(
            file_path=str(folder / "train.nadata"),
            params={"lr": "0.9"},
        )

        # The script only ran; nothing to assert on params directly, so
        # verify through a param-capturing script instead
        _write(
            folder,
            "capture.nadata",
            '#manual(v: Int = 7, name: Str = "x")\n'
            "captured = Child(nid = \"captured\", age = v, name = name);\n"
            "flush();\n",
        )

        app.execute_script_file(
            file_path=str(folder / "capture.nadata"),
            params={"v": "42", "name": "Zoe"},
        )

        captured: Any = app.database.get_by_nid(
            struct_name="Child", nid="captured"
        )

        assert captured.attributes["age"].get() == 42
        assert captured.attributes["name"].get() == "Zoe"

    def test_missing_required_errors(self, tmp_path: Any) -> None:
        """
        A required param without a value errors with the signature.
        """

        app: Nadata
        folder: Any

        app, folder = self._opened(tmp_path)

        _write(folder, "req.nadata", "#manual(who: Str)\n")

        with pytest.raises(ValueError, match="who"):
            app.execute_script_file(
                file_path=str(folder / "req.nadata")
            )

    def test_unknown_param_errors(self, tmp_path: Any) -> None:
        """
        Unknown parameter names are rejected before running.
        """

        app: Nadata
        folder: Any

        app, folder = self._opened(tmp_path)

        _write(folder, "t2.nadata", "#manual(v: Int = 1)\n")

        with pytest.raises(ValueError, match="oops"):
            app.execute_script_file(
                file_path=str(folder / "t2.nadata"),
                params={"oops": "2"},
            )

    def test_bad_coercion_errors(self, tmp_path: Any) -> None:
        """
        Malformed values are hard errors before any statement runs.
        """

        app: Nadata
        folder: Any

        app, folder = self._opened(tmp_path)

        _write(folder, "t3.nadata", "#manual(v: Int = 1)\n")

        with pytest.raises(ValueError, match="Cannot coerce"):
            app.execute_script_file(
                file_path=str(folder / "t3.nadata"),
                params={"v": "eighty"},
            )

    def test_non_manual_rejected(self, tmp_path: Any) -> None:
        """
        Explicit runs reject #init and #auto files.
        """

        app: Nadata
        folder: Any

        app, folder = self._opened(tmp_path)

        with pytest.raises(ValueError, match="not #manual"):
            app.execute_script_file(
                file_path=str(folder / "seed.nadata")
            )


class TestSessionApi:
    """
    Nadata.load integrates session folders.
    """

    def test_load_uses_pipeline(self, tmp_path: Any) -> None:
        """
        A directive folder loads via the session pipeline.
        """

        folder: Any = _project(str(tmp_path), "api")

        app: Nadata = Nadata.load(path=str(folder))

        assert app.session_folder is not None
        assert app.count("Child") == 2

    def test_legacy_folder_unchanged(self) -> None:
        """
        Directive-free example folders keep legacy behavior.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        assert app.session_folder is None
        assert app.count("Human") == 3


class TestConsoleCommands:
    """
    The console session commands.
    """

    def _console(self, tmp_path: Any) -> tuple[NadataConsole, Any]:
        """
        Opens a session and binds a console to it.

        Args:
            tmp_path (Any): Pytest tmp dir.

        Returns:
            tuple[NadataConsole, Any]: Console and folder.
        """

        folder: Any = _project(str(tmp_path), "console")

        _write(folder, "train.nadata", TRAIN)

        app: Nadata = Nadata.open_folder(folder_path=str(folder))
        console: NadataConsole = NadataConsole(app=app)

        return console, folder

    def test_run_with_params(self, tmp_path: Any) -> None:
        """
        :run executes into the persistent scope with coerced params.
        """

        console: NadataConsole
        folder: Any

        console, folder = self._console(tmp_path)

        # Quoted path: :run tokenizes with shlex, so spaces survive
        script_path: str = str(folder / "train.nadata")

        output: Any = console.execute_line(
            line=f':run "{script_path}" lr=0.5 tag=exp1'
        )

        assert "ran" in str(output)

        # Script variables landed in the persistent scope (notebook feel)
        assert console.loader.scope.get(name="lr").get() == 0.5
        assert console.loader.scope.get(name="tag").get() == "exp1"

    def test_files_listing(self, tmp_path: Any) -> None:
        """
        :files lists directives and manual signatures.
        """

        console: NadataConsole
        folder: Any

        console, folder = self._console(tmp_path)

        listing: str = str(console.execute_line(line=":files"))

        assert "#init" in listing
        assert "#auto" in listing
        assert "lr: Float" in listing

    def test_flush_command(self, tmp_path: Any) -> None:
        """
        :flush reports the written count.
        """

        console: NadataConsole
        folder: Any

        console, folder = self._console(tmp_path)

        output: str = str(console.execute_line(line=":flush"))

        assert "flushed" in output

    def test_reinit_reruns(self, tmp_path: Any) -> None:
        """
        :reinit re-runs init scripts without duplicating rows.
        """

        console: NadataConsole
        folder: Any

        console, folder = self._console(tmp_path)

        output: str = str(console.execute_line(line=":reinit seed.nadata"))

        assert "re-ran 1" in output
        assert console.app.count("Child") == 2

    def test_auto_command(self, tmp_path: Any) -> None:
        """
        :auto re-runs the auto scripts.
        """

        console: NadataConsole
        folder: Any

        console, folder = self._console(tmp_path)

        output: str = str(console.execute_line(line=":auto"))

        assert "ran 1 auto" in output
