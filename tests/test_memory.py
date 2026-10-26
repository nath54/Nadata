"""
Tests for the memory layer: record-store indexing, disk-backed lazy
rows, dirty tracking, the decode cache (LRU eviction, pinning), and
dirty-only flush write-back with removal tombstones.
"""

# Import Modules
import itertools
import os
import tempfile
from typing import Any, Optional

import pytest

from nadata.core.database.database import NaDataDatabase
from nadata.core.database.entity_id import EntityId
from nadata.core.database.persistence.binary_cells import (
    decode_batch_document,
    decode_batch_row_spans,
    encode_batch_document,
    encode_row,
)
from nadata.core.database.persistence.dumper import (
    append_instance_removal,
    dump_database,
    flush_dirty_instances,
)
from nadata.core.database.persistence.lazy_instances import (
    LazyInstanceDict,
    PendingRow,
    RowDecoder,
)
from nadata.core.database.persistence.loader import load_database
from nadata.core.database.persistence.record_store import RecordStore
from nadata.core.database.slotted_page import SlottedPage
from nadata.core.expressions.builtins import (
    ClearCacheFunction,
    PinFunction,
    UnpinFunction,
)
from nadata.core.values.values import ValueExternal, ValueTensor

_TMP = tempfile.TemporaryDirectory()


def _decoder(of: Any) -> Any:
    """
    Returns the row decoder attached to a table's lazy instance dict.

    Args:
        of (Any): The StructTable whose instances to introspect.

    Returns:
        Any: The RowDecoder instance.
    """

    return getattr(of.instances, "decoder")


def _wrap_raw_value(raw: Any) -> Any:
    """
    Wraps a raw Python value into a runtime Value for instance writes.

    Args:
        raw (Any): Raw Python value.

    Returns:
        Any: Typed runtime Value.
    """

    from nadata.api import _wrap_argument

    return _wrap_argument(value=raw)


def _make_database() -> NaDataDatabase:
    """
    Builds a small database with a Point struct and three instances.

    Returns:
        NaDataDatabase: The in-memory database.
    """

    from nadata.core.struct.struct import Struct
    from nadata.core.struct.struct_attribute import StructAttribute
    from nadata.core.types.types import TypeFloat, TypeInt

    database: NaDataDatabase = NaDataDatabase()

    struct: Struct = Struct(name="Point")
    struct.add_attribute(
        StructAttribute(name="x", attr_type=TypeFloat())
    )
    struct.add_attribute(
        StructAttribute(name="y", attr_type=TypeFloat())
    )
    struct.add_attribute(
        StructAttribute(name="w", attr_type=TypeInt())
    )

    for index in range(3):
        database.create_instance(
            struct=struct,
            attributes={
                "x": _wrap_raw_value(float(index)),
                "y": _wrap_raw_value(float(index * 10)),
                "w": _wrap_raw_value(index),
            },
        )

    return database


def _loaded() -> NaDataDatabase:
    """
    Dumps the standard test database and loads it back.

    Returns:
        NaDataDatabase: The loaded database.
    """

    path: str = _fresh_path("loaded")

    database: NaDataDatabase = _make_database()
    dump_database(database=database, storage_path=path)

    return load_database(storage_path=path)


def _fresh_path(name: str) -> str:
    """
    Returns a unique file path inside the shared temp directory.

    Args:
        name (str): File name.

    Returns:
        str: The full path.
    """

    counter: int = next(_path_counter)

    return os.path.join(_TMP.name, f"{name}_{counter}.ndb")


_path_counter = itertools.count()


class TestRecordStoreIndex:
    """
    document_locations / read_document: index-based access without
    reassembling the whole file.
    """

    def test_index_matches_read_all(self) -> None:
        """
        The index covers every document and read_document round-trips.
        """

        from nadata.core.database.page_manager import PageManager

        page_manager: PageManager = PageManager(storage_path=None)
        store: RecordStore = RecordStore(page_manager=page_manager)

        payloads: list[bytes] = [
            b"small",
            b"x" * 10_000,
            b"y" * 9_000,
            b"tiny",
        ]

        for payload in payloads:
            store.put_document(payload=payload)

        locations: dict[int, list[tuple[int, int]]] = (
            store.document_locations()
        )

        assert sorted(locations) == list(range(len(payloads)))

        all_documents: dict[int, bytes] = store.read_all_documents()

        for doc_id, expected in enumerate(payloads):
            assert store.read_document(doc_id=doc_id) == expected
            assert all_documents[doc_id] == expected

    def test_multi_page_document_locations(self) -> None:
        """
        A document spanning many pages indexes one location per chunk.
        """

        from nadata.core.database.page_manager import PageManager

        page_manager: PageManager = PageManager(storage_path=None)
        store: RecordStore = RecordStore(page_manager=page_manager)

        doc_id: int = store.put_document(payload=b"z" * 20_000)

        locations: dict[int, list[tuple[int, int]]] = (
            store.document_locations()
        )

        assert len(locations[doc_id]) > 1
        assert store.read_document(doc_id=doc_id) == b"z" * 20_000

    def test_cache_invalidated_on_write(self) -> None:
        """
        New writes appear in document_locations after a cached call.
        """

        from nadata.core.database.page_manager import PageManager

        page_manager: PageManager = PageManager(storage_path=None)
        store: RecordStore = RecordStore(page_manager=page_manager)

        store.put_document(payload=b"first")
        assert len(store.document_locations()) == 1

        second: int = store.put_document(payload=b"second")
        assert len(store.document_locations()) == 2
        assert store.read_document(doc_id=second) == b"second"

    def test_read_unknown_document(self) -> None:
        """
        Reading an unknown doc id raises KeyError.
        """

        from nadata.core.database.page_manager import PageManager

        page_manager: PageManager = PageManager(storage_path=None)
        store: RecordStore = RecordStore(page_manager=page_manager)

        with pytest.raises(KeyError):
            store.read_document(doc_id=7)

    def test_slot_count_usable_for_indexing(self) -> None:
        """
        SlottedPage exposes slot_count so indexing needs no payloads.
        """

        page: SlottedPage = SlottedPage(page_id=0)
        page.insert_record(record=b"payload")

        assert page.slot_count == 1


class TestDiskBackedRows:
    """
    Rows loaded from binary batches are doc spans, not payload copies.
    """

    def test_pending_rows_are_disk_backed(self) -> None:
        """
        Loaded pending rows carry a doc span and fetch lazily.
        """

        database: NaDataDatabase = _loaded()
        table = database.tables_by_struct_name["Point"]
        decoder: RowDecoder = _decoder(of=table)

        pending_rows: list[PendingRow] = []

        for entity_id in decoder.pending_entity_ids():

            row: Optional[PendingRow] = decoder.pending_row(
                entity_id=entity_id
            )

            assert row is not None

            pending_rows.append(row)

        pending: list[PendingRow] = pending_rows

        assert len(pending) == 3

        for row in pending:
            assert row.doc_span is not None
            assert row.doc_span[2] > 0
            assert len(row.payload) == row.doc_span[2]

    def test_all_survives_eviction_bound(self) -> None:
        """
        table.all() returns every row even when the decode-cache bound
        evicts rows mid-enumeration (full-table queries beyond the
        working-set bound used to silently drop evicted rows).
        """

        database: NaDataDatabase = _loaded()
        table = database.tables_by_struct_name["Point"]
        decoder: RowDecoder = _decoder(of=table)

        decoder.max_decoded = 1   # force eviction during enumeration

        assert len(table.all()) == 3

        texts = sorted(
            round(instance.attributes["x"].get(), 3)
            for instance in table.all()
        )

        assert texts == sorted(round(value, 3) for value in (0.0, 1.0, 2.0))

    def test_decode_from_span_matches_original(self) -> None:
        """
        Decoding a disk-backed row reproduces the original values.
        """

        database: NaDataDatabase = _loaded()
        table = database.tables_by_struct_name["Point"]

        instance = table.instances[2]

        assert instance.attributes["x"].get() == 2.0
        assert instance.attributes["y"].get() == 20.0
        assert instance.attributes["w"].get() == 2

    def test_doc_cache_bounded(self) -> None:
        """
        Row fetches keep at most the configured number of documents.
        """

        database: NaDataDatabase = _loaded()
        table = database.tables_by_struct_name["Point"]
        decoder: RowDecoder = _decoder(of=table)

        decoder._doc_cache_limit = 1

        for local_id in range(3):
            assert table.instances[local_id].attributes["x"].get() == float(
                local_id
            )

        assert len(decoder._doc_cache) <= 1

    def test_batch_row_spans_match_decode(self) -> None:
        """
        decode_batch_row_spans produces the same rows as the copy form.
        """

        rows: list[tuple[int, bytes]] = [(5, b"abc"), (9, b"de")]

        payload: bytes = encode_batch_document(
            struct_name="Point",
            columns=["x"],
            rows=rows,
            max_local_id=9,
        )

        name: str
        columns: list[str]
        max_local_id: int
        spans: list[tuple[int, int, int]]

        name, columns, max_local_id, spans = decode_batch_row_spans(
            payload=payload
        )

        assert name == "Point"
        assert columns == ["x"]
        assert max_local_id == 9
        assert [(e, payload[o:o + l]) for e, o, l in spans] == rows


class TestDirtyTracking:
    """
    Dirty flags and version snapshots drive the write-back set.
    """

    def test_loaded_instances_start_clean(self) -> None:
        """
        Decoded instances are clean until mutated.
        """

        database: NaDataDatabase = _loaded()
        table = database.tables_by_struct_name["Point"]

        assert not table.instances[0].dirty

    def test_set_attribute_marks_dirty(self) -> None:
        """
        Attribute writes flag the instance.
        """

        database: NaDataDatabase = _loaded()
        table = database.tables_by_struct_name["Point"]

        instance = table.instances[0]
        instance.set_attribute(name="x", value=_wrap_raw_value(42.0))

        assert instance.dirty

    def test_tensor_version_mutation_detected(self) -> None:
        """
        In-place tensor mutation (version bump) marks the row dirty.
        """

        from nadata.core.struct.struct import Struct
        from nadata.core.struct.struct_attribute import StructAttribute
        from nadata.core.types.types import TypeFloat, TypeTensor
        from nadata.core.values.values import make_value

        database: NaDataDatabase = NaDataDatabase()

        struct: Struct = Struct(name="Net")
        struct.add_attribute(
            StructAttribute(
                name="weights",
                attr_type=TypeTensor(value_type=TypeFloat(), dims=[2]),
            )
        )

        instance = database.create_instance(
            struct=struct,
            attributes={
                "weights": make_value(
                    value_type=TypeTensor(value_type=TypeFloat(), dims=[2]),
                    raw=[1.0, 2.0],
                )
            },
        )

        decoder: RowDecoder = RowDecoder()
        decoder.mark_clean(instance=instance)

        assert not decoder.is_dirty(instance=instance)

        weights = instance.attributes["weights"]
        assert isinstance(weights, ValueTensor)
        weights.cells[0] = 99.0
        weights.bump_version()

        assert decoder.is_dirty(instance=instance)

    def test_flush_writes_only_dirty(self) -> None:
        """
        Flush persists exactly the mutated instances.
        """

        path: str = _fresh_path("dirty")
        dump_path: str = _fresh_path("dirty")

        dump_database(database=_make_database(), storage_path=dump_path)
        database: NaDataDatabase = load_database(storage_path=dump_path)

        table = database.tables_by_struct_name["Point"]
        table.instances[1].set_attribute(name="x", value=_wrap_raw_value(123.0))

        written: int = flush_dirty_instances(
            database=database, storage_path=dump_path
        )

        assert written == 1
        assert not table.instances[1].dirty

        reloaded: NaDataDatabase = load_database(storage_path=dump_path)
        reloaded_table = reloaded.tables_by_struct_name["Point"]

        assert reloaded_table.instances[1].attributes["x"].get() == 123.0
        assert reloaded_table.instances[0].attributes["x"].get() == 0.0
        assert reloaded_table.instances[2].attributes["x"].get() == 2.0

    def test_flush_in_place_tensor_mutation(self) -> None:
        """
        A version-bumped tensor row is rewritten with the new payload.
        """

        from nadata.core.struct.struct import Struct
        from nadata.core.struct.struct_attribute import StructAttribute
        from nadata.core.types.types import TypeFloat, TypeTensor
        from nadata.core.values.values import make_value

        path: str = _fresh_path("tensor")

        source: NaDataDatabase = NaDataDatabase()

        struct: Struct = Struct(name="Net")
        struct.add_attribute(
            StructAttribute(
                name="weights",
                attr_type=TypeTensor(value_type=TypeFloat(), dims=[2]),
            )
        )

        source.create_instance(
            struct=struct,
            attributes={
                "weights": make_value(
                    value_type=TypeTensor(value_type=TypeFloat(), dims=[2]),
                    raw=[1.0, 2.0],
                )
            },
        )

        dump_database(database=source, storage_path=path)
        database: NaDataDatabase = load_database(storage_path=path)

        weights = (
            database.tables_by_struct_name["Net"]
            .instances[0]
            .attributes["weights"]
        )
        assert isinstance(weights, ValueTensor)
        weights.cells[0] = 7.5
        weights.bump_version()

        flush_dirty_instances(database=database, storage_path=path)

        reloaded: NaDataDatabase = load_database(storage_path=path)
        reloaded_weights = (
            reloaded.tables_by_struct_name["Net"]
            .instances[0]
            .attributes["weights"]
        )

        assert isinstance(reloaded_weights, ValueTensor)
        assert reloaded_weights.cells[0] == 7.5
        assert reloaded_weights.cells[1] == 2.0

    def test_runtime_instances_flush(self) -> None:
        """
        Instances created after load are dirty by default and flush.
        """

        from nadata.core.struct.struct import Struct
        from nadata.core.struct.struct_attribute import StructAttribute
        from nadata.core.types.types import TypeFloat

        path: str = _fresh_path("runtime")

        dump_database(database=_make_database(), storage_path=path)
        database: NaDataDatabase = load_database(storage_path=path)

        struct: Struct = Struct(name="Point")
        struct.add_attribute(
            StructAttribute(name="x", attr_type=TypeFloat())
        )
        struct.add_attribute(
            StructAttribute(name="y", attr_type=TypeFloat())
        )
        struct.add_attribute(
            StructAttribute(name="w", attr_type=TypeFloat())
        )

        database.create_instance(
            struct=struct,
            attributes={
                "x": _wrap_raw_value(9.0),
                "y": _wrap_raw_value(0.0),
                "w": _wrap_raw_value(0),
            },
        )

        assert flush_dirty_instances(
            database=database, storage_path=path
        ) == 1

        reloaded: NaDataDatabase = load_database(storage_path=path)
        reloaded_table = reloaded.tables_by_struct_name["Point"]

        assert reloaded_table.instances[3].attributes["x"].get() == 9.0

        # Original rows untouched
        assert reloaded_table.instances[0].attributes["x"].get() == 0.0
        assert reloaded_table.instances[1].attributes["x"].get() == 1.0
        assert reloaded_table.instances[2].attributes["x"].get() == 2.0


class TestDecodeCache:
    """
    LRU bound, eviction, and pinning.
    """

    def test_eviction_respects_bound(self) -> None:
        """
        Resolving more instances than the bound keeps the cache bounded.
        """

        path: str = _fresh_path("bound")

        dump_database(database=_make_database(), storage_path=path)
        database: NaDataDatabase = load_database(storage_path=path)

        table = database.tables_by_struct_name["Point"]
        decoder: RowDecoder = _decoder(of=table)
        decoder.max_decoded = 1

        for local_id in range(3):
            assert table.instances[local_id] is not None

        assert len(decoder.decoded) == 1

    def test_clear_cache_skips_dirty_and_pinned(self) -> None:
        """
        Only clean, unpinned instances evict.
        """

        path: str = _fresh_path("pin")

        dump_database(database=_make_database(), storage_path=path)
        database: NaDataDatabase = load_database(storage_path=path)

        table = database.tables_by_struct_name["Point"]
        decoder: RowDecoder = _decoder(of=table)

        pinned = table.instances[0]
        pinned.pinned = True

        dirty = table.instances[1]
        dirty.set_attribute(name="x", value=_wrap_raw_value(1.0))

        clean = table.instances[2]

        evicted: int = decoder.evict_clean()

        assert evicted == 1
        assert decoder.decoded.get(pinned.entity_id.value) is pinned
        assert decoder.decoded.get(dirty.entity_id.value) is dirty
        assert clean.entity_id.value not in decoder.decoded

        # Evicted instances re-decode transparently on next access
        assert table.instances[2] is not None

    def test_pin_unpin_builtins(self) -> None:
        """
        The pin/unpin builtins flip the residency flag.
        """

        path: str = _fresh_path("pinbuiltins")

        dump_database(database=_make_database(), storage_path=path)
        database: NaDataDatabase = load_database(storage_path=path)

        table = database.tables_by_struct_name["Point"]
        instance = table.instances[0]

        PinFunction().invoke(
            scope=None, args=[ValueExternal(payload=instance)], kwargs={}
        )
        assert instance.pinned

        UnpinFunction().invoke(
            scope=None, args=[ValueExternal(payload=instance)], kwargs={}
        )
        assert not instance.pinned

    def test_clear_cache_builtin_reports_count(self) -> None:
        """
        clear_cache returns the number of evicted instances.
        """

        path: str = _fresh_path("clear")

        dump_database(database=_make_database(), storage_path=path)
        database: NaDataDatabase = load_database(storage_path=path)

        table = database.tables_by_struct_name["Point"]

        for local_id in range(3):
            assert table.instances[local_id] is not None

        assert ClearCacheFunction(database=database).invoke(
            scope=None, args=[], kwargs={}
        ) == 3

    def test_flush_zero_when_memory_only(self) -> None:
        """
        A database without a file reports zero flushed instances.
        """

        assert _make_database().flush() == 0
        assert _make_database().clear_cache() == 0


class TestTombstones:
    """
    Entity removal: in-memory drop plus persisted tombstone.
    """

    def test_removal_survives_reload(self) -> None:
        """
        A tombstoned entity is gone after reload; others remain.
        """

        path: str = _fresh_path("tomb")

        database: NaDataDatabase = _make_database()
        dump_database(database=database, storage_path=path)

        loaded: NaDataDatabase = load_database(storage_path=path)
        victim = loaded.tables_by_struct_name["Point"].instances[1]

        append_instance_removal(
            database=loaded,
            storage_path=path,
            entity_id=victim.entity_id.value,
        )

        reloaded: NaDataDatabase = load_database(storage_path=path)
        reloaded_table = reloaded.tables_by_struct_name["Point"]

        with pytest.raises(KeyError):
            assert reloaded_table.instances[1]

        assert reloaded_table.instances[0] is not None
        assert reloaded_table.instances[2] is not None

    def test_ids_not_reused_after_removal(self) -> None:
        """
        New instances after a removal never take the removed local id.
        """

        from nadata.core.struct.struct import Struct
        from nadata.core.struct.struct_attribute import StructAttribute
        from nadata.core.types.types import TypeFloat

        path: str = _fresh_path("ids")

        database: NaDataDatabase = _make_database()
        dump_database(database=database, storage_path=path)

        loaded: NaDataDatabase = load_database(storage_path=path)
        victim = loaded.tables_by_struct_name["Point"].instances[1]

        append_instance_removal(
            database=loaded,
            storage_path=path,
            entity_id=victim.entity_id.value,
        )

        struct: Struct = Struct(name="Point")
        struct.add_attribute(
            StructAttribute(name="x", attr_type=TypeFloat())
        )
        struct.add_attribute(
            StructAttribute(name="y", attr_type=TypeFloat())
        )
        struct.add_attribute(
            StructAttribute(name="w", attr_type=TypeFloat())
        )

        loaded.create_instance(
            struct=struct,
            attributes={
                "x": _wrap_raw_value(5.0),
                "y": _wrap_raw_value(0.0),
                "w": _wrap_raw_value(0),
            },
        )

        loaded.flush()

        reloaded: NaDataDatabase = load_database(storage_path=path)
        reloaded_table = reloaded.tables_by_struct_name["Point"]

        with pytest.raises(KeyError):
            reloaded_table.instances[1]

        assert reloaded_table.instances[3].attributes["x"].get() == 5.0

    def test_decoder_remove_entity_clears_everything(self) -> None:
        """
        remove_entity drops pending rows, shells, and decoded entries.
        """

        decoder: RowDecoder = RowDecoder()

        row: PendingRow = PendingRow(
            entity_id=5,
            struct_name="Point",
            columns=["x"],
            types={},
            payload=b"\x00",
        )

        decoder.register(row=row)

        decoder.remove_entity(entity_id=5)

        assert not decoder.has_pending(entity_id=5)
        assert 5 not in decoder._shells

    def test_removal_then_access_redecodes_others(self) -> None:
        """
        After a tombstone, remaining rows still decode from disk.
        """

        path: str = _fresh_path("remain")

        database: NaDataDatabase = _make_database()
        dump_database(database=database, storage_path=path)

        loaded: NaDataDatabase = load_database(storage_path=path)
        table = loaded.tables_by_struct_name["Point"]
        victim = table.instances[1]

        append_instance_removal(
            database=loaded,
            storage_path=path,
            entity_id=victim.entity_id.value,
        )

        reloaded: NaDataDatabase = load_database(storage_path=path)

        assert (
            reloaded.tables_by_struct_name["Point"]
            .instances[0]
            .attributes["y"]
            .get()
            == 0.0
        )


class TestLazyDictContract:
    """
    LazyInstanceDict eviction restores pending decode.
    """

    def test_evict_entity_restores_pending(self) -> None:
        """
        After evict_entity, a fresh read re-decodes the same values.
        """

        path: str = _fresh_path("evict")

        dump_database(database=_make_database(), storage_path=path)
        database: NaDataDatabase = load_database(storage_path=path)

        table = database.tables_by_struct_name["Point"]
        decoder: RowDecoder = _decoder(of=table)

        original = table.instances[0]
        decoder._remove_decoded(entity_id=original.entity_id.value)

        assert dict.__contains__(table.instances, 0) is False

        again = table.instances[0]

        assert again is not original
        assert again.attributes["x"].get() == 0.0

    def test_evict_entity_ignores_new_instance(self) -> None:
        """
        Eviction does not drop a newer instance bound to the same key.
        """

        decoder: RowDecoder = RowDecoder()
        lazy: LazyInstanceDict = LazyInstanceDict(
            decoder=decoder, struct_name="Point"
        )

        dict.__setitem__(lazy, 0, object())

        decoder._tables["Point"] = lazy
        decoder._remove_decoded(entity_id=EntityId.from_parts(0, 0).value)

        assert dict.__contains__(lazy, 0)


class TestStreamingScale:
    """
    Acceptance: many rows, bounded decode cache, transparent streaming.
    """

    def test_hundred_k_rows_bounded_decode(self) -> None:
        """
        100k rows load with O(index) memory and decode under a bound.
        """

        import time

        path: str = _fresh_path("scale")

        started: float = time.perf_counter()

        from nadata.core.struct.struct import Struct
        from nadata.core.struct.struct_attribute import StructAttribute
        from nadata.core.types.types import TypeFloat, TypeInt

        source: NaDataDatabase = NaDataDatabase()

        struct: Struct = Struct(name="Row")
        struct.add_attribute(
            StructAttribute(name="a", attr_type=TypeFloat())
        )
        struct.add_attribute(
            StructAttribute(name="b", attr_type=TypeInt())
        )

        table: Any = source.register_struct(struct=struct)

        # Bulk insert through the table (100k rows)
        for index in range(100_000):
            table.insert(
                attributes={
                    "a": _wrap_raw_value(float(index)),
                    "b": _wrap_raw_value(index),
                }
            )

        dump_database(database=source, storage_path=path)

        database: NaDataDatabase = load_database(storage_path=path)
        loaded_table: Any = database.tables_by_struct_name["Row"]
        decoder: RowDecoder = _decoder(of=loaded_table)
        decoder.max_decoded = 1_000

        # Stream through every row: decode, read, (implicitly evict)
        total: float = 0.0
        for local_id in range(100_000):
            total += loaded_table.instances[local_id].attributes["a"].get()

        assert abs(total - 4_999_950_000.0) < 1.0
        assert len(decoder.decoded) <= 1_000

        elapsed: float = time.perf_counter() - started

        assert elapsed < 120.0


class TestScriptBuiltins:
    """
    The memory builtins work end to end from .nadata scripts.
    """

    def test_flush_pin_clear_cache_from_script(self) -> None:
        """
        A script can pin, mutate, flush, and clear the cache.
        """

        from nadata.api import Nadata

        path: str = _fresh_path("script")

        dump_database(database=_make_database(), storage_path=path)

        app: Nadata = Nadata.open(storage_path=path)

        # Bind a symbol python-side, then drive the builtins from script
        table: Any = app.database.tables_by_struct_name["Point"]
        instance: Any = table.instances[0]
        app.database.symbols["p0"] = instance.entity_id

        app.execute_data(data_text=(
            "p0.x = 77.0;\n"
            "pin(p0);\n"
            "written = flush();\n"
            "print(\"flushed:\", written);\n"
            "evicted = clear_cache();\n"
            "print(\"evicted:\", evicted);\n"
        ))

        assert not instance.dirty
        assert instance.pinned

        reloaded: NaDataDatabase = load_database(storage_path=path)

        assert (
            reloaded.tables_by_struct_name["Point"]
            .instances[0]
            .attributes["x"]
            .get()
            == 77.0
        )

    def test_remove_from_db_from_script(self) -> None:
        """
        remove_from_db drops the instance and persists the tombstone.
        """

        from nadata.api import Nadata

        path: str = _fresh_path("remove")

        dump_database(database=_make_database(), storage_path=path)

        app: Nadata = Nadata.open(storage_path=path)

        table: Any = app.database.tables_by_struct_name["Point"]
        app.database.symbols["victim"] = table.instances[1].entity_id

        app.execute_data(data_text="remove_from_db(victim);\n")

        with pytest.raises(KeyError):
            assert table.instances[1]

        reloaded: NaDataDatabase = load_database(storage_path=path)

        with pytest.raises(KeyError):
            assert reloaded.tables_by_struct_name["Point"].instances[1]

        assert (
            reloaded.tables_by_struct_name["Point"]
            .instances[0]
            .attributes["x"]
            .get()
            == 0.0
        )
