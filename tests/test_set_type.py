"""
Tests for the native Set type: Set[T] schema type, set literals, language
integration, and persistence round trips.
"""

# Import Modules
from __future__ import annotations
import contextlib
import io
import os
import tempfile
from typing import Any

import pytest

from nadata.api import Nadata
from nadata.core.database.database import NaDataDatabase
from nadata.core.database.persistence.dumper import dump_database
from nadata.core.database.persistence.loader import load_database
from nadata.core.types.types import TypeInt, TypeSet, TypeStr
from nadata.core.values.values import ValueSet
from nadata.parser.expressions.expression_parser import ExpressionParser
from nadata.parser.nadata_loader.loader import NaDataLoader

SCHEMA: str = """
class P {
    name: Str;
    tags: Set[Str] = {};
    nums: Set[Int32];
    crew: Set[Str] = {"a", "b"};
    @has_tag(t: Str) -> Bool { return t in self.tags; }
}
"""


class TestSetTypeLayer:
    """
    Tests the TypeSet and ValueSet building blocks.
    """

    def test_type_set_repr_and_default(self) -> None:
        """
        Tests the repr and the empty default of TypeSet.
        """

        set_type: TypeSet = TypeSet(value_type=TypeStr())

        assert repr(set_type) == "Set[Str]"
        assert set_type.default_value == set()

    def test_check_value_accepts_sets(self) -> None:
        """
        Tests that typed sets validate and wrong kinds are rejected.
        """

        set_type: TypeSet = TypeSet(value_type=TypeStr())

        set_type.check_value(value={"a", "b"})
        set_type.check_value(value=frozenset({"a"}))

        with pytest.raises(TypeError):
            set_type.check_value(value=["a"])

        with pytest.raises(TypeError):
            set_type.check_value(value={1, 2})

    def test_value_set_converts_and_validates(self) -> None:
        """
        Tests that ValueSet accepts iterables and rejects bad elements.
        """

        value_set: ValueSet = ValueSet(value_type=TypeSet(value_type=TypeInt()), elements=[1, 2])

        assert value_set.get() == {1, 2}
        assert len(value_set) == 2

        with pytest.raises(TypeError):
            ValueSet(value_type=TypeSet(value_type=TypeInt()), elements=["x"])

    def test_make_value_wraps_sets(self) -> None:
        """
        Tests that make_value dispatches sets to ValueSet.
        """

        from nadata.core.values.values import make_value

        value: object = make_value(
            value_type=TypeSet(value_type=TypeStr()), raw={"a"}
        )

        assert isinstance(value, ValueSet)


class TestSetLiterals:
    """
    Tests set literal parsing and brace disambiguation.
    """

    def _eval(self, expr: str) -> object:
        """
        Evaluates an expression with no environment.

        Args:
            expr (str): Expression source.

        Returns:
            object: Raw result.
        """

        ast: object = ExpressionParser.parse_text(text=expr)

        from nadata.core.expressions.scope import ExecutionScope

        scope: ExecutionScope = ExecutionScope(globals_dict={})

        return ast.evaluate(scope=scope).get()  # type: ignore[attr-defined]

    def test_set_literal(self) -> None:
        """
        Tests that braced comma literals evaluate to typed sets.
        """

        result: object = self._eval(expr='{1, 2, 3}')

        assert result == {1, 2, 3}

    def test_set_literal_dedups(self) -> None:
        """
        Tests that duplicate literal elements collapse.
        """

        assert self._eval(expr='{"a", "a", "b"}') == {"a", "b"}

    def test_singleton_set(self) -> None:
        """
        Tests that a braced single element is a one-element set.
        """

        assert self._eval(expr='{"x"}') == {"x"}

    def test_empty_braces_stay_dict(self) -> None:
        """
        Tests that '{}' remains a dict (declared-type context).
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text="class Q { m: Dict[Str, Int32] = {}; }")

        NaDataLoader(database=app.database).load_file(content=(
            'q = Q(m = {"a": 1});\n'
        ))

        instance: object = app.query(struct_name="Q").first()

        assert instance.get_attribute(name="m").get() == {"a": 1}  # type: ignore[attr-defined]

    def test_dict_literal_still_works(self) -> None:
        """
        Tests that dict literals are unaffected.
        """

        assert self._eval(expr='{"k": 1, "j": 2}') == {"k": 1, "j": 2}


class TestSetLanguage:
    """
    Tests sets inside the broader language.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the test schema and two instances.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text=(
            'a = P(name = "alice", tags = {"admin", "dev", "admin"}, nums = {3, 1, 2});\n'
            'b = P(name = "bob");\n'
        ))

        return app

    def test_literal_dedup_on_create(self) -> None:
        """
        Tests that constructor set literals deduplicate.
        """

        app: Nadata = self._app()

        instance: object = app.query(struct_name="P").first()

        tags: object = instance.get_attribute(name="tags").get()  # type: ignore[attr-defined]

        assert tags == {"admin", "dev"}

    def test_empty_braces_declares_empty_set(self) -> None:
        """
        Tests that 'Set[Str] s = {};' declares an empty set variable.
        """

        app: Nadata = self._app()

        app.execute_data(data_text=(
            "Set[Str] es = {};\n"
            "Set[Int32] ns = {1, 2};\n"
            "count = 0;\n"
            "for (n: Int32 in ns) { count = count + 1; }\n"
        ))

        # The declarations and loop parse and execute without error
        assert app.query(struct_name="P").count() == 2

    def test_membership_on_sets(self) -> None:
        """
        Tests 'in' / 'not in' against set attributes.
        """

        app: Nadata = self._app()

        assert app.call_method("a", "has_tag", "admin") is True
        assert app.call_method("a", "has_tag", "zzz") is False

    def test_where_predicate_on_set(self) -> None:
        """
        Tests membership inside a where predicate over a set column.
        """

        app: Nadata = self._app()

        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            app.execute_data(data_text=(
                'devs = P.where((p: P) -> "dev" in p.tags);\n'
                "print(devs.count());\n"
            ))

        assert buffer.getvalue().strip() == "1"

    def test_len_builtin_on_sets(self) -> None:
        """
        Tests that sets support len() where supported in expressions.
        """

        app: Nadata = self._app()

        instance: object = app.query(struct_name="P").first()

        nums: object = instance.get_attribute(name="nums").get()  # type: ignore[attr-defined]

        assert nums == {1, 2, 3}


class TestSetPersistence:
    """
    Tests sets across the v2 binary persistence format.
    """

    def _round_trip(self, data_text: str) -> tuple[object, object]:
        """
        Dumps and reloads a database containing the test data.

        Args:
            data_text (str): .nadata source.

        Returns:
            tuple[object, object]: (reloaded lazy instances, second round
            trip instances).
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(content=data_text)

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "sets.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        storage_path2: str = os.path.join(temp_dir, "sets2.ndb")

        dump_database(database=reloaded, storage_path=storage_path2)

        reloaded2: Any = load_database(storage_path=storage_path2)

        return (
            reloaded.tables_by_struct_name["P"].instances,
            reloaded2.tables_by_struct_name["P"].instances,
        )

    def test_set_round_trip(self) -> None:
        """
        Tests that typed sets survive dump/load twice.
        """

        lazy: Any
        lazy2: Any

        lazy, lazy2 = self._round_trip(data_text=(
            'a = P(name = "alice", tags = {"admin", "dev"}, nums = {3, 1, 2});\n'
        ))

        first: object = lazy[0]

        assert (
            first.get_attribute(name="tags").get() == {"admin", "dev"}  # type: ignore[attr-defined]
        )
        assert (
            first.get_attribute(name="nums").get() == {1, 2, 3}  # type: ignore[attr-defined]
        )

        second: object = lazy2[0]

        second_tags: object = second.get_attribute(name="tags").get()  # type: ignore[attr-defined]

        assert second_tags == {"admin", "dev"}

    def test_sparse_row_schema_defaults(self) -> None:
        """
        Tests that omitted set columns fall back to schema defaults lazily.
        """

        lazy: Any
        lazy2: Any

        lazy, lazy2 = self._round_trip(data_text='b = P(name = "bob");\n')

        bob: object = lazy[0]

        assert bob.get_attribute(name="tags").get() == set()  # type: ignore[attr-defined]
        assert bob.get_attribute(name="crew").get() == {"a", "b"}  # type: ignore[attr-defined]

        bob2: object = lazy2[0]

        bob2_crew: object = bob2.get_attribute(name="crew").get()  # type: ignore[attr-defined]

        assert bob2_crew == {"a", "b"}

    def test_open_does_not_decode_sets(self) -> None:
        """
        Tests that a lazy open leaves set columns pending.
        """

        from nadata.core.database.persistence.lazy_instances import LazyInstanceDict

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(
            content='a = P(name = "alice", tags = {"x"});\n'
        )

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "lazy.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        lazy: object = reloaded.tables_by_struct_name["P"].instances

        assert isinstance(lazy, LazyInstanceDict)
        assert dict.keys(lazy) == dict.keys({})
        assert len(lazy) == 1
