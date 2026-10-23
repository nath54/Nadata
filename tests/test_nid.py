"""
Tests for the identity layer: nid= named instances, partial-update
upsert with stable entity ids, lookup builtins, persistence, and
removal bookkeeping.
"""

# Import Modules
import os
import tempfile
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.persistence.dumper import flush_dirty_instances
from src.core.database.persistence.loader import load_database
from src.parser.nadata_loader.loader import NaDataLoader

_TMP = tempfile.TemporaryDirectory()


def _fresh_app() -> Nadata:
    """
    Builds an empty app over a fresh in-memory database.

    Returns:
        Nadata: The facade.
    """

    from src.core.database.database import NaDataDatabase

    return Nadata(database=NaDataDatabase())


def _child_schema() -> str:
    """
    Returns a simple Child schema source.

    Returns:
        str: The schema text.
    """

    return "class Child {\n    name: Str;\n    age: Int32 = 0;\n}\n"


def _evaluate(app: Nadata, expression: str) -> Any:
    """
    Evaluates one expression against the app's scope.

    Args:
        app (Nadata): The facade.
        expression (str): Expression source.

    Returns:
        Any: The raw result payload.
    """

    loader: NaDataLoader = NaDataLoader(database=app.database)
    loader.load_file(content="pass_marker = 1;\n")

    statements: list[Any] = loader.parse_statements(content=expression)

    result: Any = statements[0].expression.evaluate(scope=loader.scope)

    return result.get()


def _path(name: str) -> str:
    """
    Returns a unique database path in the shared temp directory.

    Args:
        name (str): File name.

    Returns:
        str: The full path.
    """

    counter: int = next(_counter)

    return os.path.join(_TMP.name, f"{name}_{counter}.ndb")


import itertools

_counter = itertools.count()


class TestNidCreation:
    """
    nid= creation and identity semantics.
    """

    def test_named_creation_and_lookup(self) -> None:
        """
        A nid'd instance is retrievable and exposes its nid.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=_child_schema())

        app.execute_data(data_text=(
            'tom = Child(nid = "tom", name = "Tom", age = 6);\n'
        ))

        tom: Any = app.database.get_by_nid(struct_name="Child", nid="tom")

        assert tom is not None
        assert tom.attributes["name"].get() == "Tom"
        assert tom.nid == "tom"

    def test_upsert_keeps_entity_id(self) -> None:
        """
        Re-declaring the same nid updates in place with a stable id.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=_child_schema())

        app.execute_data(data_text=(
            'tom = Child(nid = "tom", name = "Tom", age = 6);\n'
        ))

        first_instance: Any = app.database.get_by_nid(
            struct_name="Child", nid="tom"
        )
        assert first_instance is not None
        first_id: Any = first_instance.entity_id.value

        app.execute_data(data_text=(
            'tom = Child(nid = "tom", name = "Tommy");\n'
        ))

        second: Any = app.database.get_by_nid(
            struct_name="Child", nid="tom"
        )

        assert second.entity_id.value == first_id
        assert second.attributes["name"].get() == "Tommy"

        # Partial update: age was not passed, keeps its value
        assert second.attributes["age"].get() == 6

    def test_runtime_call_upsert(self) -> None:
        """
        nid upsert works for constructor calls inside script statements.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=_child_schema())

        app.execute_data(data_text=(
            'a = Child(nid = "k", name = "Kid", age = 5);\n'
            'b = Child(nid = "k", name = " Kiddo");\n'
        ))

        assert app.count("Child") == 1

        kid: Any = app.database.get_by_nid(struct_name="Child", nid="k")

        assert kid.attributes["name"].get() == " Kiddo"
        assert kid.attributes["age"].get() == 5

    def test_anonymous_instances_stay_anonymous(self) -> None:
        """
        Constructors without nid always create and never register.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=(
            "class Child {\n    name: Str;\n}\n"
        ))

        app.execute_data(data_text=(
            'for(i: Int32 in [1, 2, 3]) { Child(name = "kid"); }\n'
        ))

        assert app.count("Child") == 3
        assert app.database.nid_names(struct_name="Child") == []

    def test_loop_with_dynamic_nids_is_idempotent(self) -> None:
        """
        Re-running a loop with computed nids keeps every entity id.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=(
            "class Child {\n    name: Str;\n}\n"
        ))

        script: str = (
            'for(i: Int32 in [1, 2, 3]) {\n'
            '    Child(nid = "child_" + str(i), name = "kid" + str(i));\n'
            "}\n"
        )

        app.execute_data(data_text=script)

        ids_first: dict[str, Any] = {}

        for nid in ["child_1", "child_2", "child_3"]:

            found: Any = app.database.get_by_nid(
                struct_name="Child", nid=nid
            )
            assert found is not None
            ids_first[nid] = found.entity_id.value

        assert app.count("Child") == 3

        app.execute_data(data_text=script)

        assert app.count("Child") == 3

        for nid, entity_id in ids_first.items():

            again: Any = app.database.get_by_nid(
                struct_name="Child", nid=nid
            )
            assert again is not None
            assert again.entity_id.value == entity_id

    def test_nid_requires_string(self) -> None:
        """
        A non-string nid is a TypeError.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=_child_schema())

        with pytest.raises(TypeError, match="nid must be a string"):
            app.execute_data(data_text='Child(nid = 5, name = "x");\n')


class TestNidLookups:
    """
    get_by_nid / first / last / nids builtins and elt.nid.
    """

    def test_get_by_nid_missing_is_none(self) -> None:
        """
        Unknown nids resolve to a None payload, not an error.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=_child_schema())

        assert (
            _evaluate(app=app, expression='get_by_nid(Child, "nobody")')
            is None
        )

    def test_first_last_nids_builtins(self) -> None:
        """
        The convenience builtins agree with the query set.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=_child_schema())

        app.execute_data(data_text=(
            'a = Child(nid = "a", name = "A");\n'
            'b = Child(nid = "b", name = "B");\n'
            'c = Child(name = "C");\n'
        ))

        assert _evaluate(app=app, expression="nids(Child)") == ["a", "b"]
        assert (
            _evaluate(app=app, expression="first(Child)").attributes[
                "name"
            ].get()
            == "A"
        )
        assert (
            _evaluate(app=app, expression="last(Child)").attributes[
                "name"
            ].get()
            == "C"
        )
        assert (
            _evaluate(app=app, expression='get_by_nid(Child, "b").nid')
            == "b"
        )
        assert _evaluate(app=app, expression="c.nid") is None
        assert _evaluate(app=app, expression="a.id") >= 0

    def test_per_struct_namespaces(self) -> None:
        """
        The same nid may exist under different structs.
        """

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=(
            "class Child {\n    name: Str;\n}\n"
            "class Toy {\n    label: Str;\n}\n"
        ))

        app.execute_data(data_text=(
            'Child(nid = "x", name = "kid");\n'
            'Toy(nid = "x", label = "ball");\n'
        ))

        named_child: Any = app.database.get_by_nid(
            struct_name="Child", nid="x"
        )
        assert named_child is not None
        assert named_child.attributes["name"].get() == "kid"

        named_toy: Any = app.database.get_by_nid(
            struct_name="Toy", nid="x"
        )
        assert named_toy is not None
        assert named_toy.attributes["label"].get() == "ball"


class TestNidPersistence:
    """
    The nid map and instance nids survive persistence.
    """

    def _seeded_path(self, name: str) -> tuple[Nadata, str]:
        """
        Builds an app with one nid'd instance and dumps it.

        Args:
            name (str): File name.

        Returns:
            tuple[Nadata, str]: The app and the storage path.
        """

        from src.core.database.persistence.dumper import dump_database

        app: Nadata = _fresh_app()
        app.execute_schema(schema_text=_child_schema())

        app.execute_data(data_text=(
            'tom = Child(nid = "tom", name = "Tom");\n'
        ))

        path: str = _path(name)
        dump_database(database=app.database, storage_path=path)

        return app, path

    def test_dump_load_round_trip(self) -> None:
        """
        After reload, get_by_nid resolves and .nid is restored.
        """

        _, path = self._seeded_path(name="roundtrip.ndb")

        reloaded: Nadata = Nadata.open(storage_path=path)
        tom: Any = reloaded.database.get_by_nid(
            struct_name="Child", nid="tom"
        )

        assert tom is not None
        assert tom.nid == "tom"
        assert tom.attributes["name"].get() == "Tom"

    def test_flush_persists_nid_map(self) -> None:
        """
        flush() writes the nid map; a reload sees the upserted identity.
        """

        _, path = self._seeded_path(name="flush.ndb")
        reloaded: Nadata = Nadata.open(storage_path=path)

        reloaded.execute_data(data_text=(
            'tom = Child(nid = "tom", name = "Tommy");\n'
        ))

        assert (
            flush_dirty_instances(
                database=reloaded.database,
                storage_path=path,
            )
            >= 1
        )

        again: Nadata = Nadata.open(storage_path=path)
        tom: Any = again.database.get_by_nid(
            struct_name="Child", nid="tom"
        )

        assert tom.attributes["name"].get() == "Tommy"

    def test_removal_drops_nid(self) -> None:
        """
        A tombstoned instance loses its nid; re-declaration gets a new id.
        """

        _, path = self._seeded_path(name="removal.ndb")
        reloaded: Nadata = Nadata.open(storage_path=path)

        old: Any = reloaded.database.get_by_nid(
            struct_name="Child", nid="tom"
        )
        old_id: int = old.entity_id.value

        reloaded.execute_data(data_text="remove_from_db(tom);\n")

        assert reloaded.database.get_by_nid(
            struct_name="Child", nid="tom"
        ) is None

        reloaded.execute_data(data_text=(
            'tom = Child(nid = "tom", name = "Tom II");\n'
        ))

        fresh: Any = reloaded.database.get_by_nid(
            struct_name="Child", nid="tom"
        )

        assert fresh.entity_id.value != old_id
        assert fresh.attributes["name"].get() == "Tom II"

        # The removal + re-creation persist across another reload
        flush_dirty_instances(
            database=reloaded.database, storage_path=path
        )
        final: Nadata = Nadata.open(storage_path=path)

        final_tom: Any = final.database.get_by_nid(
            struct_name="Child", nid="tom"
        )
        assert final_tom is not None
        assert final_tom.entity_id.value == fresh.entity_id.value

    def test_nid_map_after_plain_load(self) -> None:
        """
        load_database restores the nid map directly (engine level).
        """

        _, path = self._seeded_path(name="engine.ndb")

        database = load_database(storage_path=path)

        assert database.get_by_nid(
            struct_name="Child", nid="tom"
        ) is not None
