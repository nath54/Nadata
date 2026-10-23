"""
Tests for the I/O builtins: text/lines/JSON/CSV round trips against
the project root, freshness helpers (exists / file_hash / settings),
and the settings persistence contract.
"""

# Import Modules
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.api import Nadata
from src.console import NadataConsole
from src.core.database.database import NaDataDatabase


SCHEMA: str = (
    "class Note {\n"
    "    title: Str;\n"
    "    score: Float64 = 0.0;\n"
    "}\n"
)

SEED: str = (
    "#init\n"
    'n = Note(nid = "first", title = "First");\n'
    "flush();\n"
)


class _Project:
    """
    Small helper owning a session folder and a bound console.
    """

    def __init__(self) -> None:
        """
        Creates the project and opens a session with a console.
        """

        self.folder: Any = Path(tempfile.mkdtemp())
        (self.folder / "schema.nastruct").write_text(SCHEMA, encoding="utf-8")
        (self.folder / "seed.nadata").write_text(SEED, encoding="utf-8")

        self.app: Nadata = Nadata.open_folder(folder_path=str(self.folder))
        self.console: NadataConsole = NadataConsole(app=self.app)

    def run(self, source: str) -> Any:
        """
        Evaluates one request in the console's persistent scope.

        Args:
            source (str): Source line(s).

        Returns:
            Any: The echoed value (None for statements).
        """

        return self.console.execute_line(line=source)

    def value(self, name: str) -> Any:
        """
        Reads a variable from the persistent scope as raw payload.

        Args:
            name (str): Variable name.

        Returns:
            Any: The raw value.
        """

        return self.console.loader.scope.get(name=name).get()

    def path(self, rel: str) -> Path:
        """
        Returns an absolute path inside the project folder.

        Args:
            rel (str): Relative path.

        Returns:
            Path: The absolute path.
        """

        resolved: Path = self.folder / rel

        return resolved

    def cleanup(self) -> None:
        """
        Removes the project folder.
        """

        shutil.rmtree(self.folder, ignore_errors=True)


class TestText:
    """
    write_text / read_text / append_text / read_lines.
    """

    def test_round_trip(self) -> None:
        """
        A written file reads back verbatim from the project root.
        """

        project: _Project = _Project()

        try:
            project.run('write_text("notes/hello.txt", "hi there");')

            assert project.path("notes/hello.txt").read_text(
                encoding="utf-8"
            ) == "hi there"

            project.run('text = read_text("notes/hello.txt");')

            assert project.value("text") == "hi there"

        finally:
            project.cleanup()

    def test_write_overwrites(self) -> None:
        """
        Writes replace the whole content (no append semantics).
        """

        project: _Project = _Project()

        try:
            project.run('write_text("f.txt", "one");')
            project.run('write_text("f.txt", "two");')

            project.run('text = read_text("f.txt");')

            assert project.value("text") == "two"

        finally:
            project.cleanup()

    def test_append_creates_and_appends(self) -> None:
        """
        append_text creates missing files and adds one line per call.
        """

        project: _Project = _Project()

        try:
            project.run('append_text("log.txt", "one");')
            project.run('append_text("log.txt", "two");')

            project.run('lines = read_lines("log.txt");')

            assert project.value("lines") == ["one", "two"]

        finally:
            project.cleanup()

    def test_read_missing_raises(self) -> None:
        """
        Reading a missing file is an error (guard with exists()).
        """

        project: _Project = _Project()

        try:
            with pytest.raises(FileNotFoundError):
                project.run('text = read_text("nope.txt");')

        finally:
            project.cleanup()

    def test_trailing_newline_no_empty_line(self) -> None:
        """
        A trailing newline does not produce an empty last entry.
        """

        project: _Project = _Project()

        try:
            project.run('write_text("f.txt", "a\\nb\\n");')

            project.run('lines = read_lines("f.txt");')

            assert project.value("lines") == ["a", "b"]

        finally:
            project.cleanup()


class TestJson:
    """
    read_json / write_json.
    """

    def test_object_mixed_numbers(self) -> None:
        """
        int/float object values share the number family (Float64).
        """

        project: _Project = _Project()

        try:
            project.run('write_json("cfg.json", {"lr": 0.1, "epochs": 3});')

            project.run('cfg = read_json("cfg.json");')

            raw: Any = project.value("cfg")

            assert raw == {"lr": 0.1, "epochs": 3}

            project.run('epochs = cfg["epochs"];')

            assert project.value("epochs") == 3

        finally:
            project.cleanup()

    def test_object_of_ints(self) -> None:
        """
        All-int objects stay Int64.
        """

        project: _Project = _Project()

        try:
            project.run('write_json("w.json", {"a": 1, "b": 2});')

            project.run('d = read_json("w.json");')

            project.run('sum = d["a"] + d["b"];')

            assert project.value("sum") == 3

        finally:
            project.cleanup()

    def test_array_round_trip(self) -> None:
        """
        Arrays of scalars round-trip.
        """

        project: _Project = _Project()

        try:
            project.run('write_json("arr.json", [10, 20, 30]);')

            project.run('nums = read_json("arr.json");')

            assert project.value("nums") == [10, 20, 30]

        finally:
            project.cleanup()

    def test_mixed_object_raises(self) -> None:
        """
        Mixed scalar families are a hard error with guidance.
        """

        project: _Project = _Project()

        try:
            project.path("mixed.json").write_text(
                json.dumps({"tag": "exp1", "lr": 0.5}), encoding="utf-8"
            )

            with pytest.raises(ValueError, match="same scalar family"):
                project.run('d = read_json("mixed.json");')

        finally:
            project.cleanup()

    def test_nested_raises(self) -> None:
        """
        Nested structures are rejected (keep JSON flat).
        """

        project: _Project = _Project()

        try:
            project.path("nested.json").write_text(
                json.dumps({"a": {"b": 1}}), encoding="utf-8"
            )

            with pytest.raises(ValueError, match="nested"):
                project.run('d = read_json("nested.json");')

        finally:
            project.cleanup()


class TestCsv:
    """
    read_csv / write_csv — stringly-typed by design.
    """

    def test_round_trip(self) -> None:
        """
        Rows round-trip as dict[Str, Str] with the header preserved.
        """

        project: _Project = _Project()

        try:
            project.run(
                'rows = [{"name": "a", "val": "1"}, '
                '{"name": "b", "val": "2"}];'
            )

            project.run('written = write_csv("out.csv", rows);')

            assert project.value("written") == 2

            project.run('back = read_csv("out.csv");')

            assert project.value("back") == [
                {"name": "a", "val": "1"},
                {"name": "b", "val": "2"},
            ]

        finally:
            project.cleanup()

    def test_no_inference(self) -> None:
        """
        Numeric-looking cells stay strings.
        """

        project: _Project = _Project()

        try:
            project.run('rows = [{"id": "007", "score": "1.50"}];')

            project.run('write_csv("t.csv", rows);')

            project.run('back = read_csv("t.csv");')

            assert project.value("back") == [{"id": "007", "score": "1.50"}]

        finally:
            project.cleanup()

    def test_ragged_row_raises(self) -> None:
        """
        A row shorter than the header is a hard error.
        """

        project: _Project = _Project()

        try:
            project.path("ragged.csv").write_text(
                "a,b\n1\n", encoding="utf-8"
            )

            with pytest.raises(ValueError, match="cells"):
                project.run('rows = read_csv("ragged.csv");')

        finally:
            project.cleanup()

    def test_key_mismatch_raises(self) -> None:
        """
        Rows must share the first row's key set.
        """

        project: _Project = _Project()

        try:
            project.run('rows = [{"a": "1"}, {"b": "2"}];')

            with pytest.raises(ValueError, match="keys"):
                project.run('write_csv("bad.csv", rows);')

        finally:
            project.cleanup()


class TestFreshness:
    """
    exists / file_hash / list_dir / get_setting / set_setting.
    """

    def test_exists(self) -> None:
        """
        exists() reflects the filesystem.
        """

        project: _Project = _Project()

        try:
            project.run('ok = exists("seed.nadata");')

            assert project.value("ok") is True

            project.run('ok = exists("ghost.txt");')

            assert project.value("ok") is False

        finally:
            project.cleanup()

    def test_file_hash_detects_changes(self) -> None:
        """
        Identical contents hash equal; edits change the hash.
        """

        project: _Project = _Project()

        try:
            project.run('write_text("d.txt", "payload");')

            project.run('h1 = file_hash("d.txt");')

            project.run('write_text("d.txt", "payload!");')

            project.run('h2 = file_hash("d.txt");')

            assert project.value("h1") != project.value("h2")

        finally:
            project.cleanup()

    def test_list_dir_sorted(self) -> None:
        """
        list_dir returns sorted entry names.
        """

        project: _Project = _Project()

        try:
            project.run('names = list_dir(".");')

            names: Any = project.value("names")

            assert names == sorted(names)
            assert "seed.nadata" in names

        finally:
            project.cleanup()

    def test_settings_round_trip_and_persist(self) -> None:
        """
        set_setting/get_setting round-trip and survive reopen+flush.
        """

        project: _Project = _Project()

        try:
            project.run(
                'write_text("input.csv", "x\\n1\\n");'
            )

            project.run(
                'set_setting("data_hash", file_hash("input.csv"));'
            )

            project.run("flush();")

            # Reopen the world: the marker survives
            reloaded: Nadata = Nadata.open_folder(
                folder_path=str(project.folder)
            )

            console: NadataConsole = NadataConsole(app=reloaded)

            console.execute_line(line='h = get_setting("data_hash");')

            assert console.loader.scope.get(name="h").get() == (
                console.loader.scope.get(name="__missing__").get()
                if False
                else reloaded.database.settings["data_hash"]
            )

        finally:
            project.cleanup()

    def test_get_setting_missing_is_none(self) -> None:
        """
        Unknown settings read as None (missing is not an error).
        """

        project: _Project = _Project()

        try:
            project.run('v = get_setting("never_set");')

            assert project.console.loader.scope.get(name="v") is None or (
                project.value("v") is None
            )

        finally:
            project.cleanup()


class TestConversions:
    """
    int() / float() — the explicit bridge from stringly-typed CSV.
    """

    def test_csv_to_float(self) -> None:
        """
        CSV cells convert explicitly at the use site.
        """

        project: _Project = _Project()

        try:
            project.run('rows = [{"sensor": "a", "value": "21.5"}];')

            project.run('write_csv("in.csv", rows);')

            project.run('data = read_csv("in.csv");')

            project.run('x = float(data[0]["value"]) + 1.5;')

            assert project.value("x") == 23.0

            project.run('n = int("21") + 1;')

            assert project.value("n") == 22

        finally:
            project.cleanup()

    def test_malformed_raises(self) -> None:
        """
        Malformed numbers are hard errors (no silent zeros).
        """

        project: _Project = _Project()

        try:
            with pytest.raises(ValueError):
                project.run('x = float("twenty");')

            with pytest.raises(ValueError):
                project.run('y = int("1.5");')

        finally:
            project.cleanup()


class TestMissingComparisons:
    """
    None payloads (missing get_setting / get_by_nid) in comparisons.
    """

    def test_missing_setting_compare(self) -> None:
        """
        A missing setting is != anything and == nothing; the freshness
        idiom works on the first run.
        """

        project: _Project = _Project()

        try:
            project.run('write_text("in.txt", "payload");')

            # First run: get_setting is missing -> != holds
            project.run(
                'changed = get_setting("data_hash") '
                '!= file_hash("in.txt");'
            )

            assert project.value("changed") is True

            project.run('set_setting("data_hash", file_hash("in.txt"));')

            # Second run: equal -> no change
            project.run(
                'changed = get_setting("data_hash") '
                '!= file_hash("in.txt");'
            )

            assert project.value("changed") is False

        finally:
            project.cleanup()


class TestPathResolution:
    """
    io_base rules: project root, CWD fallback, absolute passthrough.
    """

    def test_absolute_path_passthrough(self) -> None:
        """
        Absolute paths bypass the project root.
        """

        tmp: Path = Path(tempfile.mkdtemp())

        project: _Project = _Project()

        try:
            target: Path = tmp / "abs.txt"

            project.run(f'write_text("{target.as_posix()}", "abs");')

            assert target.read_text(encoding="utf-8") == "abs"

        finally:
            project.cleanup()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cwd_fallback_without_session(self) -> None:
        """
        Without a project root, paths resolve against the CWD.
        """

        tmp: Path = Path(tempfile.mkdtemp())
        old_cwd: str = os.getcwd()

        try:
            os.chdir(tmp)

            app: Nadata = Nadata(database=NaDataDatabase())

            app.execute_data('write_text("cwd.txt", "here");')

            assert (tmp / "cwd.txt").read_text(encoding="utf-8") == "here"

            assert app.database.io_base is None

        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmp, ignore_errors=True)


class TestSessionAcceptance:
    """
    The quiz scenario: a #manual pipeline driven by I/O builtins.
    """

    def test_config_driven_manual(self) -> None:
        """
        A #manual script reads its config from the project root and
        writes results back; the world stays independent per --db.
        """

        folder: Any = Path(tempfile.mkdtemp())
        (folder / "schema.nastruct").write_text(SCHEMA, encoding="utf-8")
        (folder / "seed.nadata").write_text(SEED, encoding="utf-8")

        (folder / "config.json").write_text(
            json.dumps({"score": 42.0, "bonus": 2.0}), encoding="utf-8"
        )

        (folder / "score.nadata").write_text(
            "#manual(target: Str = \"first\", tag: Str = \"exp1\")\n"
            "cfg = read_json(\"config.json\");\n"
            "n = get_by_nid(Note, target);\n"
            "n.score = cfg[\"score\"] + cfg[\"bonus\"];\n"
            "write_json(\"results.json\", {\"scored\": target});\n"
            "flush();\n",
            encoding="utf-8",
        )

        try:
            app: Nadata = Nadata.open_folder(folder_path=str(folder))

            app.execute_script_file(file_path="score.nadata")

            assert json.loads(
                (folder / "results.json").read_text(encoding="utf-8")
            ) == {"scored": "first"}

            # The instance was updated through the file config
            scored: Any = app.database.get_by_nid(
                struct_name="Note", nid="first"
            )

            assert scored is not None
            assert scored.get_attribute(name="score").get() == 44.0

            # A second world starts fresh (no results.json inside)
            app2: Nadata = Nadata.open_folder(
                folder_path=str(folder), db_path="worlds/second.ndb"
            )

            assert not (folder / "worlds" / "results.json").exists()
            fresh: Any = app2.database.get_by_nid(
                struct_name="Note", nid="first"
            )

            assert fresh is not None
            assert fresh.get_attribute(name="score").get() == 0.0

        finally:
            shutil.rmtree(folder, ignore_errors=True)
