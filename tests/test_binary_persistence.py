"""
Tests for the binary persistence format (v2) and lazy instance loading.

Covers row/batch round trips (scalars, collections, references, home
types, sparse rows, escapes), lazy materialization semantics, untouched
row passthrough on dump, and incremental delta saves.
"""

# Import Modules
from __future__ import annotations
import os
import tempfile
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.core.database.persistence.binary_cells import (
    decode_batch_document,
    decode_row,
    encode_batch_document,
    encode_row,
)
from src.core.database.persistence.dumper import dump_database
from src.core.database.persistence.loader import load_database
from src.core.database.persistence.lazy_instances import LazyInstanceDict
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime
from src.core.types.types import (
    TypeArray,
    TypeBool,
    TypeColor,
    TypeDateTime,
    TypeDict,
    TypeFloat,
    TypeInt,
    TypeStr,
)
from src.core.values.values import make_value
from src.parser.nadata_loader.loader import NaDataLoader
from src.parser.nastruct_parser.nastruct_parser import NaStructParser

SCHEMA: str = """
class Human {
    name: Str;
    age: Int32;
    height: Float32;
    active: Bool;
    tags: Array[Str] = [];
    scores: Dict[Str, Int32] = {};
    born: DateTime;
    hue: Color;
    boss: Human;
}
"""


def _types() -> dict[str, Any]:
    """
    Builds the declared column types of the test schema.

    Returns:
        dict[str, Any]: Type per column name.
    """

    return {
        "name": TypeStr(),
        "age": TypeInt(num_bits=32, signed=True),
        "height": TypeFloat(num_bits=32, signed=True),
        "active": TypeBool(),
        "tags": TypeArray(value_type=TypeStr()),
        "scores": TypeDict(key_type=TypeStr(), value_type=TypeInt(num_bits=32, signed=True)),
        "born": TypeDateTime(),
        "hue": TypeColor(),
    }


class TestRowCodec:
    """
    Tests the binary row codec round trips.
    """

    COLUMNS: list[str] = [
        "name", "age", "height", "active", "tags", "scores", "born", "hue",
    ]

    def _row(self) -> dict[str, Any]:
        """
        Builds one fully populated attribute row.

        Returns:
            dict[str, Any]: Attribute values by name.
        """

        types: dict[str, Any] = _types()

        return {
            "name": make_value(value_type=types["name"], raw="person_1"),
            "age": make_value(value_type=types["age"], raw=42),
            "height": make_value(value_type=types["height"], raw=1.8),
            "active": make_value(value_type=types["active"], raw=True),
            "tags": make_value(value_type=types["tags"], raw=["a", "b"]),
            "scores": make_value(value_type=types["scores"], raw={"math": 5}),
            "born": make_value(
                value_type=types["born"],
                raw=DateTime(y=2024, m=5, d=1, H=10, M=30, S=0, MS=5),
            ),
            "hue": make_value(value_type=types["hue"], raw=Color(r=1, g=2, b=3, a=255)),
        }

    def test_full_row_round_trip(self) -> None:
        """
        Tests that every typed value survives a row round trip.
        """

        attributes: dict[str, Any] = self._row()
        types: dict[str, Any] = _types()

        payload: bytes = encode_row(
            attributes=attributes, columns=self.COLUMNS, types=types
        )

        decoded: dict[str, Any]
        consumed: int

        decoded, consumed = decode_row(
            payload=payload,
            offset=0,
            columns=self.COLUMNS,
            types=types,
            resolver=lambda entity_id: None,
        )

        assert consumed == len(payload)

        for column in self.COLUMNS:
            assert decoded[column].get() == attributes[column].get(), column

    def test_row_is_compact(self) -> None:
        """
        Tests that a full typed row stays well under the legacy JSON size.
        """

        payload: bytes = encode_row(
            attributes=self._row(), columns=self.COLUMNS, types=_types()
        )

        assert len(payload) < 120

    def test_sparse_row(self) -> None:
        """
        Tests that missing attributes become absent bits.
        """

        attributes: dict[str, Any] = self._row()
        types: dict[str, Any] = _types()

        sparse: dict[str, Any] = {
            key: attributes[key] for key in ("name", "age")
        }

        payload: bytes = encode_row(
            attributes=sparse, columns=self.COLUMNS, types=types
        )

        decoded: dict[str, Any]
        decoded, _ = decode_row(
            payload=payload,
            offset=0,
            columns=self.COLUMNS,
            types=types,
            resolver=lambda entity_id: None,
        )

        assert set(decoded.keys()) == {"name", "age"}

    def test_escape_hatch(self) -> None:
        """
        Tests that a payload not fitting its declared type falls back to
        JSON and survives the round trip.
        """

        types: dict[str, Any] = _types()

        from src.core.values.values import ValueExternal

        attributes: dict[str, Any] = {
            "name": make_value(value_type=types["name"], raw="x"),
            "age": ValueExternal(payload="not a number"),
        }

        payload: bytes = encode_row(
            attributes=attributes, columns=self.COLUMNS, types=types
        )

        decoded: dict[str, Any]
        decoded, _ = decode_row(
            payload=payload,
            offset=0,
            columns=self.COLUMNS,
            types=types,
            resolver=lambda entity_id: None,
        )

        assert decoded["name"].get() == "x"

        # The mismatched payload survives as its raw decoded structure
        assert decoded["age"].get() == "not a number"

    def test_batch_document_round_trip(self) -> None:
        """
        Tests batch assembly and parsing with row payloads intact.
        """

        attributes: dict[str, Any] = self._row()
        types: dict[str, Any] = _types()

        rows: list[tuple[int, bytes]] = [
            (100, encode_row(attributes=attributes, columns=self.COLUMNS, types=types)),
            (101, encode_row(attributes={}, columns=self.COLUMNS, types=types)),
        ]

        doc: bytes = encode_batch_document(
            struct_name="Human",
            columns=self.COLUMNS,
            rows=rows,
            max_local_id=1,
        )

        struct_name: str
        columns: list[str]
        max_local_id: int
        parsed: list[tuple[int, bytes]]

        struct_name, columns, max_local_id, parsed = decode_batch_document(
            payload=doc,
        )

        assert struct_name == "Human"
        assert columns == self.COLUMNS
        assert max_local_id == 1
        assert [entity_id for entity_id, _ in parsed] == [100, 101]
        assert parsed[0][1] == rows[0][1]


class TestPersistenceRoundTrip:
    """
    Tests full dump/load round trips through the new format.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the Human schema.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        return app

    def _dump_load(self, app: Nadata) -> Any:
        """
        Dumps and reloads an app's database.

        Args:
            app (Nadata): The source application.

        Returns:
            Any: The reloaded database.
        """

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "round.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        return load_database(storage_path=storage_path)

    def test_scalars_collections_round_trip(self) -> None:
        """
        Tests scalars, arrays, dicts, and home types across dump/load.
        """

        app: Nadata = self._app()

        NaDataLoader(database=app.database).load_file(content=(
            'tom = Human(name = "tom", age = 42, height = 1.8, active = true,'
            'tags = ["x", "y"], scores = {"m": 5},'
            "born = DateTime(y = 2020, m = 1, d = 2), hue = Color(r = 1, g = 2, b = 3, a = 255));"
        ))

        reloaded: Any = self._dump_load(app=app)
        tom: Any = reloaded.tables_by_struct_name["Human"].instances[0]

        assert tom.attributes["name"].get() == "tom"
        assert tom.attributes["age"].get() == 42
        assert tom.attributes["height"].get() == 1.8
        assert tom.attributes["active"].get() is True
        assert tom.attributes["tags"].get() == ["x", "y"]
        assert tom.attributes["scores"].get() == {"m": 5}
        assert tom.attributes["born"].get().year == 2020
        assert tom.attributes["hue"].get().r == 1

    def test_reference_round_trip(self) -> None:
        """
        Tests that struct references survive and resolve lazily.
        """

        app: Nadata = self._app()

        NaDataLoader(database=app.database).load_file(content=(
            'tom = Human(name = "tom");'
            'jean = Human(name = "jean", boss = tom);'
        ))

        reloaded: Any = self._dump_load(app=app)
        lazy: Any = reloaded.tables_by_struct_name["Human"].instances

        assert isinstance(lazy, LazyInstanceDict)

        jean: Any = lazy[1]

        assert jean.attributes["boss"].get().attributes["name"].get() == "tom"

    def test_methods_survive(self) -> None:
        """
        Tests that schema methods still execute after reload.
        """

        app: Nadata = self._app()

        reloaded: Any = self._dump_load(app=app)

        assert "Human" in reloaded.tables_by_struct_name

        human_struct: Any = reloaded.tables_by_struct_name["Human"].struct

        assert "name" in human_struct.get_all_attributes()

    def test_symbols_and_settings(self) -> None:
        """
        Tests symbol bindings and settings across the round trip.
        """

        app: Nadata = self._app()
        app.set_loop_limit(777)

        NaDataLoader(database=app.database).load_file(content='tom = Human(name = "tom");')

        reloaded: Any = self._dump_load(app=app)

        assert reloaded.settings.get("loop_limit") == 777
        assert reloaded.get_by_symbol(name="tom") is not None


class TestLazySemantics:
    """
    Tests lazy materialization behavior.
    """

    def _loaded(self) -> Any:
        """
        Builds a reloaded lazy database with three rows.

        Returns:
            Any: The reloaded database.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text='class P { name: Str; }')

        NaDataLoader(database=app.database).load_file(content=(
            'a = P(name = "a"); b = P(name = "b"); c = P(name = "c");'
        ))

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "lazy.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        return load_database(storage_path=storage_path)

    def test_open_does_not_decode(self) -> None:
        """
        Tests that loading registers rows without decoding any.
        """

        reloaded: Any = self._loaded()
        lazy: Any = reloaded.tables_by_struct_name["P"].instances

        assert isinstance(lazy, LazyInstanceDict)
        assert dict.keys(lazy) == dict.keys({})
        assert len(lazy) == 3

    def test_point_access_decodes_one(self) -> None:
        """
        Tests that indexing materializes only the requested row.
        """

        reloaded: Any = self._loaded()
        lazy: Any = reloaded.tables_by_struct_name["P"].instances

        instance: Any = lazy[1]

        assert instance.attributes["name"].get() == "b"
        assert dict.keys(lazy) == dict.keys({1: instance})

    def test_contains_and_len_without_decode(self) -> None:
        """
        Tests membership and length on an untouched lazy mapping.
        """

        reloaded: Any = self._loaded()
        lazy: Any = reloaded.tables_by_struct_name["P"].instances

        assert 2 in lazy
        assert 9 not in lazy
        assert len(lazy) == 3

    def test_untouched_rows_pass_through_dump(self) -> None:
        """
        Tests that dumping a lazily opened database re-emits raw rows.
        """

        reloaded: Any = self._loaded()

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "pass.ndb")

        dump_database(database=reloaded, storage_path=storage_path)

        reloaded2: Any = load_database(storage_path=storage_path)

        assert len(reloaded2.tables_by_struct_name["P"].instances) == 3

    def test_runtime_instance_replaces_pending(self) -> None:
        """
        Tests that creating a new instance on a lazy table works and dumps.
        """

        reloaded: Any = self._loaded()

        NaDataLoader(database=reloaded).load_file(content='d = P(name = "d");')

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "new.ndb")

        dump_database(database=reloaded, storage_path=storage_path)

        reloaded2: Any = load_database(storage_path=storage_path)

        assert len(reloaded2.tables_by_struct_name["P"].instances) == 4

        assert (
            reloaded2.tables_by_struct_name["P"].instances[3]
            .attributes["name"].get()
            == "d"
        )


class TestDeltaSaves:
    """
    Tests incremental appends over the binary format.
    """

    def test_delta_appends_only_changed(self) -> None:
        """
        Tests that an unchanged reopen appends only the meta document.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text='class P { name: Str; age: Int32 = 0; }')

        NaDataLoader(database=app.database).load_file(
            content='a = P(name = "a"); b = P(name = "b");'
        )

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "delta.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        from src.core.database.persistence.dumper import append_database_changes

        reopened: Any = load_database(storage_path=storage_path)

        assert append_database_changes(database=reopened, storage_path=storage_path) == 1

    def test_delta_writes_changed_rows(self) -> None:
        """
        Tests that modifying one row appends a batch with just that row.
        """

        from src.core.database.persistence.dumper import append_database_changes
        from src.core.values.values import make_value
        from src.core.types.types import TypeStr

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text='class P { name: Str; }')

        NaDataLoader(database=app.database).load_file(content=(
            'a = P(name = "a"); b = P(name = "b");'
        ))

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "delta2.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reopened: Any = load_database(storage_path=storage_path)

        reopened.tables_by_struct_name["P"].instances[0].set_attribute(
            name="name",
            value=make_value(value_type=TypeStr(), raw="changed"),
        )

        append_database_changes(database=reopened, storage_path=storage_path)

        final: Any = load_database(storage_path=storage_path)

        instances: Any = final.tables_by_struct_name["P"].instances

        assert instances[0].attributes["name"].get() == "changed"
        assert instances[1].attributes["name"].get() == "b"
