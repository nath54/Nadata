"""
Tests for the persistence round trip.

Covers the RecordStore chunking over slotted pages, and full database
dump / load cycles preserving structs, AST method bodies, functions,
entity ids, cross instance references, and symbols.
"""

# Import Modules
import os
from typing import Any

from src.core.expressions.expression import LiteralExpr
from src.core.types.types import TypeArray
from src.core.values.values import Value, ValueArray

from src.api import Nadata
from src.core.database.persistence import (
    append_database_changes,
    compact_database,
    count_documents,
    dump_database,
    load_database,
)
from src.core.database.persistence.record_store import RecordStore
from src.core.database.page_manager import PageManager
from src.core.database.database import NaDataDatabase
from src.parser.nadata_loader.loader import NaDataLoader
from src.parser.nastruct_parser.nastruct_parser import NaStructParser
from src.parser.nastruct_parser.resolver import build_type_globals


def load_example_database(example_name: str) -> NaDataDatabase:
    """
    Parses and loads a full example folder (structs and data).

    Args:
        example_name (str): Example folder name under ./examples.

    Returns:
        NaDataDatabase: The loaded database.
    """

    parser: NaStructParser = NaStructParser()
    database: NaDataDatabase = parser.parse_folder(
        folder_path=f"./examples/{example_name}"
    )

    loader: NaDataLoader = NaDataLoader(database=database)
    loader.load_folder(folder_path=f"./examples/{example_name}/data")

    return database


class TestRecordStore:
    """
    Tests chunked document storage over slotted pages.
    """

    def test_small_document_round_trip(self, tmp_path: Any) -> None:
        """
        Tests that a small document survives a store and read cycle.
        """

        page_manager: PageManager = PageManager(
            storage_path=str(tmp_path / "store.bin")
        )
        record_store: RecordStore = RecordStore(page_manager=page_manager)

        doc_id: int = record_store.put_document(payload=b"hello nadata")
        page_manager.flush()

        reader_page_manager: PageManager = PageManager(
            storage_path=str(tmp_path / "store.bin")
        )
        reader: RecordStore = RecordStore(page_manager=reader_page_manager)

        assert reader.read_all_documents() == {doc_id: b"hello nadata"}

    def test_large_document_chunking(self, tmp_path: Any) -> None:
        """
        Tests that a document larger than a page is chunked and reassembled.
        """

        page_manager: PageManager = PageManager(
            storage_path=str(tmp_path / "store.bin")
        )
        record_store: RecordStore = RecordStore(page_manager=page_manager)

        large_payload: bytes = bytes(range(256)) * 600

        doc_id: int = record_store.put_document(payload=large_payload)
        page_manager.flush()

        reader_page_manager: PageManager = PageManager(
            storage_path=str(tmp_path / "store.bin")
        )
        reader: RecordStore = RecordStore(page_manager=reader_page_manager)

        assert reader.read_all_documents() == {doc_id: large_payload}

    def test_multiple_documents_order(self, tmp_path: Any) -> None:
        """
        Tests that several documents come back keyed by their doc ids.
        """

        page_manager: PageManager = PageManager(
            storage_path=str(tmp_path / "store.bin")
        )
        record_store: RecordStore = RecordStore(page_manager=page_manager)

        record_store.put_document(payload=b"first")
        record_store.put_document(payload=b"second")
        record_store.put_document(payload=b"third")
        page_manager.flush()

        reader_page_manager: PageManager = PageManager(
            storage_path=str(tmp_path / "store.bin")
        )
        reader: RecordStore = RecordStore(page_manager=reader_page_manager)

        documents: dict[int, bytes] = reader.read_all_documents()

        assert list(documents.values()) == [b"first", b"second", b"third"]


class TestPersistenceRoundTrip:
    """
    Tests full database dump and load cycles on the examples.
    """

    def test_ex0_structs_functions_symbols(self, tmp_path: Any) -> None:
        """
        Tests that schemas, functions, and symbols survive a round trip.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "ex0.bin")

        dump_database(database=database, storage_path=storage_path)
        loaded: NaDataDatabase = load_database(storage_path=storage_path)

        assert sorted(loaded.tables_by_struct_name) == sorted(
            database.tables_by_struct_name
        )
        assert sorted(loaded.functions) == sorted(database.functions)
        assert sorted(loaded.symbols) == sorted(database.symbols)

        # Inheritance chain preserved
        cat_struct: Any = loaded.tables_by_struct_name["Cat"].struct
        entity_struct: Any = loaded.tables_by_struct_name["Entity"].struct

        assert cat_struct.is_subclass_of(other=entity_struct)

    def test_ex0_instances_and_defaults(self, tmp_path: Any) -> None:
        """
        Tests that instance attributes and schema defaults survive.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "ex0.bin")

        dump_database(database=database, storage_path=storage_path)
        loaded: NaDataDatabase = load_database(storage_path=storage_path)

        tom: Any = loaded.get_by_symbol(name="tom")

        assert tom.get_attribute("name").get() == "Tom"
        assert tom.get_attribute("gender").get() == "m"

        # Explicitly set attribute and inherited array default
        assert tom.get_attribute("birthplace").get() == "Paris"
        assert tom.get_attribute("children").get() == []

        # Cross instance references
        parent_names: list[str] = [
            parent.get_attribute("name").get()
            for parent in tom.get_attribute("parents").get()
        ]

        assert parent_names == ["Jean", "Louise"]

    def test_ex0_method_execution_after_load(self, tmp_path: Any) -> None:
        """
        Tests that decoded AST method bodies execute after a round trip.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "ex0.bin")

        dump_database(database=database, storage_path=storage_path)
        loaded: NaDataDatabase = load_database(storage_path=storage_path)

        tom: Any = loaded.get_by_symbol(name="tom")
        globals_dict: dict[str, Any] = build_type_globals()

        is_adult: Any = loaded.functions["is_adult"]

        assert is_adult.execute(
            self_instance=None,
            args=[tom],
            globals_dict=globals_dict
        ).get() is True

    def test_ex2_circular_references(self, tmp_path: Any) -> None:
        """
        Tests that circular instance references survive a round trip.
        """

        database: NaDataDatabase = load_example_database("ex2_inheritance")
        storage_path: str = str(tmp_path / "ex2.bin")

        dump_database(database=database, storage_path=storage_path)
        loaded: NaDataDatabase = load_database(storage_path=storage_path)

        db_primary: Any = loaded.get_by_symbol(name="db_primary")
        replica_backup: Any = loaded.get_by_symbol(name="replica_backup")

        primary_peers: Any = db_primary.get_attribute("connected_peers")
        replica_peers: Any = replica_backup.get_attribute("connected_peers")

        assert replica_backup in primary_peers.get()
        assert db_primary in replica_peers.get()

    def test_ex2_method_with_argument_after_load(self, tmp_path: Any) -> None:
        """
        Tests that methods with arguments execute on the loaded database.
        """

        from src.core.types.types import Float64
        from src.core.values.values import ValueFloat

        database: NaDataDatabase = load_example_database("ex2_inheritance")
        storage_path: str = str(tmp_path / "ex2.bin")

        dump_database(database=database, storage_path=storage_path)
        loaded: NaDataDatabase = load_database(storage_path=storage_path)

        db_primary: Any = loaded.get_by_symbol(name="db_primary")
        globals_dict: dict[str, Any] = build_type_globals()

        result: Any = db_primary.call_method(
            name="is_overloaded",
            args=[],
            kwargs={
                "current_load": ValueFloat(value_type=Float64, value=0.92)
            },
            globals_dict=globals_dict,
        )

        assert result.get() is True

    def test_entity_ids_preserved(self, tmp_path: Any) -> None:
        """
        Tests that entity ids are identical before and after the round trip.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "ex0.bin")

        original_ids: dict[str, int] = {
            name: entity_id.value for name, entity_id in database.symbols.items()
        }

        dump_database(database=database, storage_path=storage_path)
        loaded: NaDataDatabase = load_database(storage_path=storage_path)

        loaded_ids: dict[str, int] = {
            name: entity_id.value for name, entity_id in loaded.symbols.items()
        }

        assert loaded_ids == original_ids

        tom: Any = loaded.get_by_symbol(name="tom")

        assert tom.entity_id.value == original_ids["tom"]

    def test_ex3_home_type_values(self, tmp_path: Any) -> None:
        """
        Tests that DateTime, Color, and dict entries survive a round trip.
        """

        database: NaDataDatabase = load_example_database("ex3_collections")
        storage_path: str = str(tmp_path / "ex3.bin")

        dump_database(database=database, storage_path=storage_path)
        loaded: NaDataDatabase = load_database(storage_path=storage_path)

        theme: Any = loaded.get_by_symbol(name="theme_dark")

        assert theme.get_attribute("name").get() == "Cyberpunk_Dark"

        primary_color: Any = theme.get_attribute("primary_color")

        assert repr(primary_color.get()) == "Color(15, 15, 25)"

        color_map: Any = theme.get_attribute("color_map")

        assert repr(color_map.get()["background"]) == "Color(15, 15, 25)"

    def test_double_round_trip(self, tmp_path: Any) -> None:
        """
        Tests that dumping a loaded database produces an equivalent database.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        first_path: str = str(tmp_path / "first.bin")
        second_path: str = str(tmp_path / "second.bin")

        dump_database(database=database, storage_path=first_path)
        loaded: NaDataDatabase = load_database(storage_path=first_path)
        dump_database(database=loaded, storage_path=second_path)
        reloaded: NaDataDatabase = load_database(storage_path=second_path)

        assert sorted(reloaded.symbols) == sorted(database.symbols)
        assert sorted(reloaded.functions) == sorted(database.functions)

        tom: Any = reloaded.get_by_symbol(name="tom")

        assert tom.get_attribute("name").get() == "Tom"

    def test_dump_existing_path_raises(self, tmp_path: Any) -> None:
        """
        Tests that dumping over an existing file raises ValueError.
        """

        import pytest

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "exists.bin")

        storage_path_file = open(storage_path, "w")
        storage_path_file.close()

        with pytest.raises(ValueError, match="already exists"):
            dump_database(database=database, storage_path=storage_path)

    def test_load_invalid_file_raises(self, tmp_path: Any) -> None:
        """
        Tests that loading a non database file raises ValueError.
        """

        import pytest

        invalid_path: str = str(tmp_path / "invalid.bin")

        invalid_file = open(invalid_path, "w")
        invalid_file.write("not a database")
        invalid_file.close()

        with pytest.raises(ValueError, match="not a Nadata database dump"):
            load_database(storage_path=invalid_path)


def _wrap_raw_value(raw: Any) -> Value:
    """
    Wraps a raw Python value into a runtime Value for direct instance writes.

    Delegates to the API wrapper which handles scalars, struct instances,
    and lists of those.

    Args:
        raw (Any): Raw Python value.

    Returns:
        Value: Typed runtime Value.
    """

    from src.api import _wrap_argument

    return _wrap_argument(value=raw)


class TestIncrementalPersistence:
    """
    Tests appending deltas to existing database files.
    """

    def test_save_changes_after_mutation(self, tmp_path: Any) -> None:
        """
        Tests that new and changed instances are appended and reloaded.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "incr.ndb")

        dump_database(database=database, storage_path=storage_path)

        reopened: NaDataDatabase = load_database(storage_path=storage_path)
        reopened.create_instance(
            struct=reopened.tables_by_struct_name["Human"].struct,
            symbol="late_guy",
        )
        tom: Any = reopened.get_by_symbol(name="tom")
        tom.set_attribute(name="birthplace", value=_wrap_raw_value("Lyon"))

        append_database_changes(database=reopened, storage_path=storage_path)

        final: NaDataDatabase = load_database(storage_path=storage_path)

        assert final.get_by_symbol(name="late_guy") is not None

        final_tom: Any = final.get_by_symbol(name="tom")
        final_jean: Any = final.get_by_symbol(name="jean")

        assert final_tom.get_attribute("birthplace").get() == "Lyon"
        assert final_jean.get_attribute("name").get() == "Jean"

    def test_save_changes_twice(self, tmp_path: Any) -> None:
        """
        Tests that repeated incremental saves keep accumulating correctly.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "incr.ndb")

        dump_database(database=database, storage_path=storage_path)

        first: NaDataDatabase = load_database(storage_path=storage_path)
        first.create_instance(
            struct=first.tables_by_struct_name["Human"].struct,
            symbol="guy_one",
        )
        append_database_changes(database=first, storage_path=storage_path)

        second: NaDataDatabase = load_database(storage_path=storage_path)
        second.create_instance(
            struct=second.tables_by_struct_name["Human"].struct,
            symbol="guy_two",
        )
        append_database_changes(database=second, storage_path=storage_path)

        final: NaDataDatabase = load_database(storage_path=storage_path)

        assert final.get_by_symbol(name="guy_one") is not None
        assert final.get_by_symbol(name="guy_two") is not None
        assert len(final.symbols) == len(database.symbols) + 2

    def test_save_changes_without_changes(self, tmp_path: Any) -> None:
        """
        Tests that saving an unchanged database appends only the meta doc.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "incr.ndb")

        dump_database(database=database, storage_path=storage_path)
        size_before: int = os.path.getsize(storage_path)

        reopened: NaDataDatabase = load_database(storage_path=storage_path)
        written: int = append_database_changes(
            database=reopened, storage_path=storage_path
        )

        assert written == 1

        final: NaDataDatabase = load_database(storage_path=storage_path)

        assert sorted(final.symbols) == sorted(database.symbols)
        assert os.path.getsize(storage_path) >= size_before

    def test_save_changes_reference_update(self, tmp_path: Any) -> None:
        """
        Tests that changed instance references are detected and appended.
        """

        database: NaDataDatabase = load_example_database("ex0_simple")
        storage_path: str = str(tmp_path / "incr.ndb")

        dump_database(database=database, storage_path=storage_path)

        reopened: NaDataDatabase = load_database(storage_path=storage_path)
        jean: Any = reopened.get_by_symbol(name="jean")
        louise: Any = reopened.get_by_symbol(name="louise")

        jean.set_attribute(name="parents", value=_wrap_raw_value([louise]))

        append_database_changes(database=reopened, storage_path=storage_path)

        final: NaDataDatabase = load_database(storage_path=storage_path)

        final_jean: Any = final.get_by_symbol(name="jean")
        final_parents: Any = final_jean.get_attribute("parents").get()

        assert len(final_parents) == 1
        assert final_parents[0].entity_id.value == louise.entity_id.value

    def test_append_to_missing_file_raises(self, tmp_path: Any) -> None:
        """
        Tests that appending to a missing file raises ValueError.
        """

        import pytest

        database: NaDataDatabase = load_example_database("ex0_simple")

        with pytest.raises(ValueError, match="does not exist"):
            append_database_changes(
                database=database,
                storage_path=str(tmp_path / "missing.ndb"),
            )

    def test_incremental_with_executed_data(self, tmp_path: Any) -> None:
        """
        Tests the full pipeline: dump, execute data text, append, reload.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        storage_path: str = str(tmp_path / "pipeline.ndb")

        app.dump(storage_path=storage_path)

        reopened: Nadata = Nadata.open(storage_path=storage_path)
        reopened.execute_data(data_text='bob = Human(name = "Bob")')
        reopened.save_changes(storage_path=storage_path)

        final: Nadata = Nadata.open(storage_path=storage_path)

        assert final.get_attribute(symbol="bob", name="name") == "Bob"
        assert final.count("Human") == 4


class TestCompaction:
    """
    Tests rewriting a database file into a minimal snapshot.
    """

    def _build_history(self, storage_path: str) -> None:
        """
        Creates a database file with several appended deltas.

        Args:
            storage_path (str): Target .ndb path (must not exist).
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        app.dump(storage_path=storage_path)

        opened: Nadata = Nadata.open(storage_path=storage_path)
        opened.execute_data(data_text='zoe = Human(name = "Zoe")')
        opened.set_attribute(symbol="tom", name="birthplace", value="Lyon")
        opened.save_changes(storage_path=storage_path)

        opened.execute_data(data_text='bob = Human(name = "Bob")')
        opened.save_changes(storage_path=storage_path)

    def test_compact_in_place_reduces_documents(self, tmp_path: Any) -> None:
        """
        Tests that in-place compaction shrinks the document count.
        """

        storage_path: str = str(tmp_path / "db.ndb")

        self._build_history(storage_path=storage_path)
        before: int = count_documents(storage_path=storage_path)

        compact_database(storage_path=storage_path)
        after: int = count_documents(storage_path=storage_path)

        assert after < before

    def test_compact_preserves_state(self, tmp_path: Any) -> None:
        """
        Tests that the compacted file holds the exact same logical state.
        """

        storage_path: str = str(tmp_path / "db.ndb")

        self._build_history(storage_path=storage_path)
        compact_database(storage_path=storage_path)

        final: Nadata = Nadata.open(storage_path=storage_path)

        assert "zoe" in final.symbol_names
        assert "bob" in final.symbol_names
        assert final.get_attribute(symbol="tom", name="birthplace") == "Lyon"
        assert final.get_attribute(symbol="zoe", name="name") == "Zoe"
        assert final.count("Human") == 5

        # Methods still execute from the reloaded file
        assert final.call_method("tom", "older", final.get("jean")) is False

    def test_compact_to_new_file_keeps_source(self, tmp_path: Any) -> None:
        """
        Tests that compacting to another path leaves the source untouched.
        """

        storage_path: str = str(tmp_path / "db.ndb")
        output_path: str = str(tmp_path / "compact.ndb")

        self._build_history(storage_path=storage_path)
        before: int = count_documents(storage_path=storage_path)

        compact_database(storage_path=storage_path, output_path=output_path)

        assert count_documents(storage_path=storage_path) == before
        assert count_documents(storage_path=output_path) < before

        final: Nadata = Nadata.open(storage_path=output_path)

        assert final.get_attribute(symbol="zoe", name="name") == "Zoe"

    def test_compact_missing_file_raises(self, tmp_path: Any) -> None:
        """
        Tests that compacting a missing file raises ValueError.
        """

        import pytest

        with pytest.raises(ValueError, match="does not exist"):
            compact_database(storage_path=str(tmp_path / "missing.ndb"))

    def test_compact_equals_fresh_dump(self, tmp_path: Any) -> None:
        """
        Tests that the compacted document count matches a fresh full dump
        of the same logical state.
        """

        storage_path: str = str(tmp_path / "db.ndb")
        reference_path: str = str(tmp_path / "reference.ndb")

        self._build_history(storage_path=storage_path)
        compact_database(storage_path=storage_path)

        app: Nadata = Nadata.open(storage_path=storage_path)

        # Re-dump the reloaded state: compaction must be idempotent
        app.dump(storage_path=reference_path)

        assert count_documents(storage_path=storage_path) == count_documents(
            storage_path=reference_path
        )

    def test_facade_compact_in_place_and_continue(self, tmp_path: Any) -> None:
        """
        Tests the facade in-place compaction, including continued use.
        """

        storage_path: str = str(tmp_path / "db.ndb")

        self._build_history(storage_path=storage_path)

        app: Nadata = Nadata.open(storage_path=storage_path)
        before: int
        after: int
        before, after = app.compact()

        assert after < before

        # The facade reloaded: subsequent operations hit the fresh file
        app.execute_data(data_text='carl = Human(name = "Carl")')
        app.save_changes(storage_path=storage_path)

        final: Nadata = Nadata.open(storage_path=storage_path)

        assert final.get_attribute(symbol="carl", name="name") == "Carl"

    def test_facade_compact_without_file_raises(self) -> None:
        """
        Tests that compacting a project-loaded facade requires a path.
        """

        import pytest

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        with pytest.raises(ValueError, match="No database file to compact"):
            app.compact()
