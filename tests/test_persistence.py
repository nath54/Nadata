"""
Tests for the persistence round trip.

Covers the RecordStore chunking over slotted pages, and full database
dump / load cycles preserving structs, AST method bodies, functions,
entity ids, cross instance references, and symbols.
"""

# Import Modules
from typing import Any

from src.core.database.persistence import dump_database, load_database
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
