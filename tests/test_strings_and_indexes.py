"""
Tests for the string member library (upper, lower, strip, split, join,
replace, contains, starts_with, ends_with) and the hash index layer
(index / get_by with insert, update, and removal maintenance).
"""

# Import Modules
from typing import Any

import pytest

from nadata.api import Nadata
from nadata.core.database.database import NaDataDatabase


def _app() -> Any:
    """
    Creates a fresh Nadata facade over an empty database.

    Returns:
        Any: The Nadata instance.
    """

    return Nadata(database=NaDataDatabase())


class TestStringMembers:
    """
    The typed string manipulation library.
    """

    def _exec(self, script: str) -> Any:
        """
        Executes data source on a fresh app.

        Args:
            script (str): .nadata source.

        Returns:
            Any: The app.
        """

        app: Any = _app()
        app.execute_data(script)

        return app

    def test_case_members(self) -> None:
        """
        upper, lower, and strip chain and return Str.
        """

        self._exec(
            's: Str = "  Hi  ";'
            'print("a:", s.strip().upper(), "b:", s.strip().lower());'
        )

    def test_split_returns_str_array(self) -> None:
        """
        split produces a typed Array[Str].
        """

        app: Any = self._exec(
            'parts: Array[Str] = "a,b,c".split(",");'
            'print("count:", parts.length(), "first:", parts[0]);'
        )

    def test_join(self) -> None:
        """
        join concatenates an Array[Str] with the receiver separator.
        """

        self._exec(
            'parts: Array[Str] = ["x", "y"];'
            'print("joined:", "-".join(parts));'
        )

    def test_join_rejects_non_str_array(self) -> None:
        """
        join rejects arrays of non-Str elements.
        """

        app: Any = _app()

        with pytest.raises(TypeError):
            app.execute_data('"-".join([1, 2]);')

    def test_contains_starts_ends(self) -> None:
        """
        The three Bool probes behave and validate argument types.
        """

        self._exec(
            's: Str = "hello world";'
            'print(s.contains("lo wo"), s.starts_with("hell"), s.ends_with("rld"));'
        )

    def test_replace(self) -> None:
        """
        replace substitutes all occurrences.
        """

        self._exec('print("aaa".replace("a", "b"));')

    def test_wrong_argument_type_rejected(self) -> None:
        """
        String members reject non-Str arguments with a TypeError.
        """

        app: Any = _app()

        with pytest.raises(TypeError):
            app.execute_data('"abc".contains(5);')

    def test_unknown_member_rejected(self) -> None:
        """
        Unknown members do not leak Python str methods.
        """

        app: Any = _app()

        with pytest.raises(AttributeError):
            app.execute_data('"abc".title();')


SCHEMA: str = (
    "class User {\n"
    "    email: Str;\n"
    "    score: Int32 = 0;\n"
    "}\n"
)


class TestHashIndexes:
    """
    index() / get_by() with insert, update, and removal maintenance.
    """

    def _indexed_app(self) -> Any:
        """
        Creates an app with a schema, one row, and indexes built.

        Returns:
            Any: The Nadata instance.
        """

        app: Any = _app()
        app.execute_schema(schema_text=SCHEMA)
        app.execute_data('User(email = "a@x.io", score = 42);')

        return app

    def test_index_requires_declared_attribute(self) -> None:
        """
        Indexing an undeclared attribute raises AttributeError.
        """

        app: Any = self._indexed_app()

        with pytest.raises(AttributeError):
            app.execute_data('index(User, "nope");')

    def test_index_rejects_non_scalar(self) -> None:
        """
        Indexing a non-scalar attribute raises TypeError.
        """

        app: Any = self._indexed_app()
        app.execute_schema(
            "class Holder {\n"
            "    tags: Array[Str];\n"
            "}\n"
        )

        with pytest.raises(TypeError):
            app.execute_data('index(Holder, "tags");')

    def test_get_by_finds_row(self) -> None:
        """
        get_by resolves a row through the index.
        """

        app: Any = self._indexed_app()
        app.execute_data('index(User, "email");')

        table: Any = app.database.tables_by_struct_name["User"]

        found: Any = table.lookup_by_index(attribute="email", raw="a@x.io")

        assert found is not None and len(found) == 1
        assert found[0].get_attribute(name="score").get() == 42

    def test_get_by_missing_returns_empty(self) -> None:
        """
        A lookup miss returns an empty result, not an error.
        """

        app: Any = self._indexed_app()
        app.execute_data('index(User, "email");')

        table: Any = app.database.tables_by_struct_name["User"]

        assert table.lookup_by_index(attribute="email", raw="zz") == []

    def test_update_maintains_index(self) -> None:
        """
        Rewriting an indexed attribute moves the row between buckets.
        """

        app: Any = self._indexed_app()
        app.execute_data(
            'index(User, "email");'
            'u = get_by(User, "email", "a@x.io");'
            'u.email = "b@x.io";'
        )

        table: Any = app.database.tables_by_struct_name["User"]

        assert table.lookup_by_index(attribute="email", raw="a@x.io") == []
        assert len(table.lookup_by_index(attribute="email", raw="b@x.io")) == 1

    def test_insert_joins_index(self) -> None:
        """
        Rows created after index construction are indexed too.
        """

        app: Any = self._indexed_app()
        app.execute_data(
            'index(User, "email");'
            'User(email = "c@x.io", score = 7);'
        )

        table: Any = app.database.tables_by_struct_name["User"]

        found: Any = table.lookup_by_index(attribute="email", raw="c@x.io")

        assert found is not None and len(found) == 1
        assert found[0].get_attribute(name="score").get() == 7

    def test_default_value_indexed(self) -> None:
        """
        Rows relying on schema defaults are indexed under the default.
        """

        app: Any = self._indexed_app()
        app.execute_data('index(User, "score");')

        table: Any = app.database.tables_by_struct_name["User"]

        found: Any = table.lookup_by_index(attribute="score", raw=42)

        assert found is not None and len(found) == 1

    def test_removal_drops_from_index(self) -> None:
        """
        Removing an instance removes it from every index.
        """

        app: Any = self._indexed_app()
        app.execute_data('index(User, "email"); index(User, "score");')

        database: Any = app.database
        table: Any = database.tables_by_struct_name["User"]
        instance: Any = table.get(local_id=0)

        database.remove_instance(instance=instance)

        assert table.lookup_by_index(attribute="email", raw="a@x.io") == []
        assert table.lookup_by_index(attribute="score", raw=42) == []

    def test_unindexed_lookup_falls_back_to_scan(self) -> None:
        """
        get_by on an unindexed attribute scans and still finds rows.
        """

        app: Any = self._indexed_app()

        # No index() call: the language builtin scans
        table: Any = app.database.tables_by_struct_name["User"]

        assert table.lookup_by_index(attribute="email", raw="a@x.io") is None

        app.execute_data('u = get_by(User, "email", "a@x.io");')

    def test_get_by_language_surface(self) -> None:
        """
        The get_by builtin resolves and exposes the instance.
        """

        app: Any = self._indexed_app()
        app.execute_data(
            'index(User, "email");'
            'u = get_by(User, "email", "a@x.io");'
            'print("score:", u.score);'
        )
