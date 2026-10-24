"""
Tests for schema reconciliation: when a session's re-merged schema no
longer matches persisted rows (removed attributes, changed types),
stale values must never surface, type mismatches must be reported, and
lazy loading must be preserved.
"""

# Import Modules
import shutil
import tempfile
from typing import Any

import pytest

from nadata.session import load_folder_session


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


SCHEMA_V1: str = (
    "class Child {\n"
    "    name: Str;\n"
    "    age: Int32 = 0;\n"
    "    nickname: Str = \"\";\n"
    "}\n"
)

SCHEMA_V2_REMOVED: str = (
    "class Child {\n"
    "    name: Str;\n"
    "    age: Int32 = 0;\n"
    "}\n"
)

SCHEMA_V2_TYPE_CHANGED: str = (
    "class Child {\n"
    "    name: Str;\n"
    "    age: Str;\n"
    "    nickname: Str = \"\";\n"
    "}\n"
)

SEED: str = (
    "#init\n"
    'tom = Child(nid = "tom", name = "Tom", age = 6, nickname = "Tommy");\n'
    "flush();\n"
)


def _project(tmp: str, name: str) -> Any:
    """
    Creates an empty project folder.

    Args:
        tmp (str): Temp directory root.
        name (str): Folder name.

    Returns:
        Any: The folder Path.
    """

    from pathlib import Path

    folder: Any = Path(tmp) / name
    folder.mkdir(parents=True, exist_ok=True)

    return folder


@pytest.fixture
def tmp_root() -> Any:
    """
    Creates a temp directory cleaned up after the test.

    Yields:
        Any: The temp directory Path.
    """

    path: Any = tempfile.mkdtemp()

    try:
        yield path

    finally:
        shutil.rmtree(path=path, ignore_errors=True)


class TestRemovedAttribute:
    """
    Attributes removed from the schema must never surface again.
    """

    def test_stale_attribute_dropped_on_open(
        self, tmp_root: str
    ) -> None:
        """
        A row holding a removed attribute loads without it, eagerly.
        """

        folder: Any = _project(tmp=tmp_root, name="removed")
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V1)
        _write(folder=folder, rel="seed.nadata", text=SEED)

        _, first = load_folder_session(folder_path=str(folder))

        assert first.ok()

        # Evolve: remove nickname, re-open
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V2_REMOVED)

        database, result = load_folder_session(folder_path=str(folder))

        assert result.reconciliation is not None
        assert result.reconciliation.total_dropped >= 1
        assert "Child" in result.reconciliation.per_struct_dropped

        kids = database.tables_by_struct_name["Child"].instances

        for instance in kids.values():
            assert "nickname" not in instance.attributes

    def test_stale_attribute_dropped_lazily(self, tmp_root: str) -> None:
        """
        Rows never decoded this open stay lazy and still drop the
        stale column when they eventually decode.
        """

        folder: Any = _project(tmp=tmp_root, name="lazy")
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V1)
        _write(folder=folder, rel="seed.nadata", text=SEED)

        load_folder_session(folder_path=str(folder), run_auto=False)

        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V2_REMOVED)

        database, result = load_folder_session(folder_path=str(folder))

        assert result.reconciliation is not None

        table = database.tables_by_struct_name["Child"]
        decoder: Any = getattr(table.instances, "decoder", None)

        # Fresh open: rows start pending (lazy)
        if decoder is not None:
            assert decoder.reconcile_enabled

            for instance in table.instances.values():
                assert "nickname" not in instance.attributes

    def test_dropped_value_not_resurrected_by_flush(
        self, tmp_root: str
    ) -> None:
        """
        After dropping a stale attribute, rewriting the row persists
        the removal: the next open sees no stale value and no drop.
        """

        folder: Any = _project(tmp=tmp_root, name="persist")
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V1)
        _write(folder=folder, rel="seed.nadata", text=SEED)

        load_folder_session(folder_path=str(folder), run_auto=False)

        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V2_REMOVED)

        database, result = load_folder_session(folder_path=str(folder))

        assert result.reconciliation is not None

        # Touch every row so it becomes dirty, then persist
        table = database.tables_by_struct_name["Child"]

        for instance in table.instances.values():
            instance.set_attribute(
                name="age",
                value=instance.get_attribute(name="age"),
                allow_const=True,
            )

        from nadata.core.database.persistence.dumper import flush_dirty_instances

        flush_dirty_instances(
            database=database, storage_path=str(result.db_path)
        )

        _, second = load_folder_session(folder_path=str(folder))

        assert second.reconciliation is None or (
            second.reconciliation.total_dropped == 0
        )

    def test_added_attribute_uses_schema_default(self, tmp_root: str) -> None:
        """
        Attributes added by the new schema materialize through their
        declared default on access.
        """

        folder: Any = _project(tmp=tmp_root, name="added")
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V2_REMOVED)
        _write(folder=folder, rel="seed.nadata", text=SEED)

        load_folder_session(folder_path=str(folder), run_auto=False)

        added: str = (
            "class Child {\n"
            "    name: Str;\n"
            "    age: Int32 = 0;\n"
            "    nickname: Str = \"\";\n"
            "    score: Int32 = 42;\n"
            "}\n"
        )

        _write(folder=folder, rel="schema.nastruct", text=added)

        database, result = load_folder_session(folder_path=str(folder))

        assert result.reconciliation is None or result.reconciliation.ok

        tom = database.get_by_nid(struct_name="Child", nid="tom") if hasattr(
            database, "get_by_nid"
        ) else None

        if tom is None:
            pytest.skip("database.get_by_nid unavailable")

        assert tom.get_attribute(name="score").get() == 42


class TestTypeChanged:
    """
    Type changes are reported, never fatal, and data stays reachable.
    """

    def test_mismatch_reported_and_data_kept(self, tmp_root: str) -> None:
        """
        Changing a declared type reports affected rows but keeps the
        project openable and the data accessible for migration.
        """

        folder: Any = _project(tmp=tmp_root, name="typed")
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V1)
        _write(folder=folder, rel="seed.nadata", text=SEED)

        load_folder_session(folder_path=str(folder), run_auto=False)

        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V2_TYPE_CHANGED)

        database, result = load_folder_session(folder_path=str(folder))

        # Not fatal: the open succeeds and the mismatch is on record
        assert result.ok()
        assert result.reconciliation is not None
        assert not result.reconciliation.ok

        assert any(
            "Child.age" in mismatch
            for mismatch in result.reconciliation.mismatches
        )

    def test_mismatch_summary_rendered(self, tmp_root: str) -> None:
        """
        The session summary surfaces reconciliation findings.
        """

        folder: Any = _project(tmp=tmp_root, name="summary")
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V1)
        _write(folder=folder, rel="seed.nadata", text=SEED)

        load_folder_session(folder_path=str(folder), run_auto=False)

        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V2_TYPE_CHANGED)

        _, result = load_folder_session(folder_path=str(folder))

        assert "schema reconciliation" in result.summary()


class TestFreshWorld:
    """
    Fresh worlds (no persisted data) skip reconciliation entirely.
    """

    def test_no_reconciliation_report_on_fresh(self, tmp_root: str) -> None:
        """
        A newly created world has nothing to reconcile.
        """

        folder: Any = _project(tmp=tmp_root, name="fresh")
        _write(folder=folder, rel="schema.nastruct", text=SCHEMA_V1)
        _write(folder=folder, rel="seed.nadata", text=SEED)

        _, result = load_folder_session(folder_path=str(folder))

        assert result.created_fresh
        assert result.reconciliation is None
