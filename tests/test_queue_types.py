"""
Tests for the native Queue and PriorityQueue types.

Covers the type/value layers, typed member calls (push/pop/peek/size),
language integration (for-in, 'in', methods), lazy default
materialization, priority type restrictions, and persistence round trips.
"""

# Import Modules
from __future__ import annotations
import os
import tempfile
from typing import Any

import pytest

from nadata.api import Nadata
from nadata.core.database.database import NaDataDatabase
from nadata.core.database.persistence.dumper import dump_database
from nadata.core.database.persistence.loader import load_database
from nadata.core.types.types import (
    TypeFloat,
    TypeInt,
    TypePriorityQueue,
    TypeQueue,
    TypeStr,
)
from nadata.core.values.values import (
    ValuePriorityQueue,
    ValueQueue,
    make_value,
)
from nadata.parser.nadata_loader.loader import NaDataLoader
from nadata.parser.expressions.type_parser import TypeParser
from nadata.parser.nastruct_parser.resolver import TypeResolver

SCHEMA: str = """
class T {
    name: Str;
    jobs: Queue[Str] = {};
    pq: PriorityQueue[Str] = {};
    pqf: PriorityQueue[Str, Float32] = {};
    @next_job() -> Str { return self.jobs.pop(); }
    @next_urgent() -> Str { return self.pq.pop(); }
}
"""


class TestQueueTypes:
    """
    Tests the TypeQueue and TypePriorityQueue descriptors.
    """

    def test_reprs(self) -> None:
        """
        Tests the type reprs.
        """

        assert repr(TypeQueue(value_type=TypeStr())) == "Queue[Str]"
        assert repr(
            TypePriorityQueue(value_type=TypeStr(), priority_type=TypeInt())
        ) == "PriorityQueue[Str, Int32]"

    def test_priority_type_restriction(self) -> None:
        """
        Tests that only basic IntXX / FloatXX priorities are accepted.
        """

        # Basic numerics are accepted
        TypePriorityQueue(value_type=TypeStr(), priority_type=TypeInt())
        TypePriorityQueue(value_type=TypeStr(), priority_type=TypeFloat())

        # Strings are rejected
        with pytest.raises(TypeError):
            TypePriorityQueue(value_type=TypeStr(), priority_type=TypeStr())

    def test_generic_resolution(self) -> None:
        """
        Tests generic resolution of Queue[T] and PriorityQueue[T, P].
        """

        resolver: TypeResolver = TypeResolver()

        queue_type: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="Queue[Str]")
        )

        assert isinstance(queue_type, TypeQueue)

        pq_type: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="PriorityQueue[Str, Int32]")
        )

        assert isinstance(pq_type, TypePriorityQueue)

        # One-arg form defaults the priority to Int64
        pq_default: Any = resolver.resolve(
            tmp_type=TypeParser.parse_text(text="PriorityQueue[Str]")
        )

        assert isinstance(pq_default, TypePriorityQueue)
        assert pq_default.priority_type == TypeInt(num_bits=64, signed=True) or (
            getattr(pq_default.priority_type, "num_bits", None) == 64
            and getattr(pq_default.priority_type, "signed", None) is True
        )

class TestQueueValues:
    """
    Tests the ValueQueue and ValuePriorityQueue runtime values.
    """

    def test_queue_converts_iterables(self) -> None:
        """
        Tests that ValueQueue converts lists (codec path) to a deque.
        """

        queue: ValueQueue = ValueQueue(
            value_type=TypeQueue(value_type=TypeStr()), elements=["a", "b"]
        )

        assert list(queue.get()) == ["a", "b"]
        assert len(queue) == 2

    def test_queue_validates_elements(self) -> None:
        """
        Tests that queue elements are validated against the element type.
        """

        with pytest.raises(TypeError):
            ValueQueue(value_type=TypeQueue(value_type=TypeStr()), elements=[1])

    def test_priority_queue_normalizes_order(self) -> None:
        """
        Tests that entries are normalized priority ascending regardless of
        insertion order.
        """

        queue: ValuePriorityQueue = ValuePriorityQueue(
            value_type=TypePriorityQueue(value_type=TypeStr(), priority_type=TypeInt()),
            entries=[(5, "low"), (1, "high")],
        )

        assert queue.get() == [(1, "high"), (5, "low")]

    def test_priority_queue_push_pop_peek(self) -> None:
        """
        Tests push / pop / peek semantics with tie stability.
        """

        queue: ValuePriorityQueue = ValuePriorityQueue(
            value_type=TypePriorityQueue(value_type=TypeStr(), priority_type=TypeInt()),
            entries=[],
        )

        queue.push(element="b", priority=2)
        queue.push(element="a", priority=1)
        queue.push(element="c", priority=2)

        assert queue.peek() == "a"
        assert queue.pop() == "a"
        assert queue.pop() == "b"
        assert queue.pop() == "c"

        with pytest.raises(IndexError):
            queue.pop()

    def test_priority_queue_validates_priority(self) -> None:
        """
        Tests that push validates the priority against the priority type.
        """

        queue: ValuePriorityQueue = ValuePriorityQueue(
            value_type=TypePriorityQueue(value_type=TypeStr(), priority_type=TypeInt()),
            entries=[],
        )

        with pytest.raises(TypeError):
            queue.push(element="x", priority="high")

    def test_make_value_dispatch(self) -> None:
        """
        Tests that make_value wraps raw payloads for both queue types.
        """

        queue: Any = make_value(
            value_type=TypeQueue(value_type=TypeStr()), raw=["a"]
        )

        assert isinstance(queue, ValueQueue)

        pq: Any = make_value(
            value_type=TypePriorityQueue(value_type=TypeStr(), priority_type=TypeInt()),
            raw=[(1, "a")],
        )

        assert isinstance(pq, ValuePriorityQueue)


class TestQueueLanguage:
    """
    Tests queues inside the broader language.
    """

    def _app(self) -> Nadata:
        """
        Builds an app with the test schema and one instance.

        Returns:
            Nadata: The prepared facade.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text=(
            't = T(name = "worker", jobs = ["j1", "j2", "j3"]);\n'
            't.pq.push("cleanup", 5);\n'
            't.pq.push("fire", 1);\n'
            't.pqf.push("x", 1.5);\n'
        ))

        return app

    def test_constructor_list_coerces(self) -> None:
        """
        Tests that an array literal coerces into a typed FIFO queue.
        """

        app: Nadata = self._app()

        instance: Any = app.query(struct_name="T").first()

        jobs: Any = instance.get_attribute(name="jobs")

        assert isinstance(jobs, ValueQueue)
        assert list(jobs.get()) == ["j1", "j2", "j3"]

    def test_typed_push_pop_peek(self) -> None:
        """
        Tests typed member calls through the language.
        """

        app: Nadata = self._app()

        assert app.call_method("t", "next_job") == "j1"
        assert app.call_method("t", "next_urgent") == "fire"

        instance: Any = app.query(struct_name="T").first()

        assert list(instance.get_attribute(name="jobs").get()) == ["j2", "j3"]

    def test_push_validates_against_schema(self) -> None:
        """
        Tests that pushing a wrongly typed priority raises TypeError.
        """

        app: Nadata = self._app()

        instance: Any = app.query(struct_name="T").first()

        with pytest.raises(TypeError):
            instance.get_attribute(name="pq").push(
                element="x", priority="high"
            )

    def test_for_in_over_queues(self) -> None:
        """
        Tests for-in iteration over both queue kinds.
        """

        app: Nadata = self._app()

        buffer: Any = __import__("io").StringIO()

        import contextlib

        with contextlib.redirect_stdout(buffer):
            app.execute_data(data_text=(
                "count = 0;\n"
                "for (j: Str in t.jobs) { count = count + 1; }\n"
                'print("fifo:", count);\n'
                "pcount = 0;\n"
                "for (j: Str in t.pq) { pcount = pcount + 1; }\n"
                'print("pq:", pcount);\n'
            ))

        assert "fifo: 3" in buffer.getvalue()
        assert "pq: 2" in buffer.getvalue()

    def test_membership_on_queues(self) -> None:
        """
        Tests 'in' probes elements for both queue kinds.
        """

        app: Nadata = self._app()

        buffer: Any = __import__("io").StringIO()

        import contextlib

        with contextlib.redirect_stdout(buffer):
            app.execute_data(data_text=(
                'print("j1" in t.jobs);\n'
                'print("zz" in t.jobs);\n'
                'print("fire" in t.pq);\n'
                'print("zz" in t.pq);\n'
            ))

        assert buffer.getvalue().split() == ["True", "False", "True", "False"]

    def test_unmaterialized_default_push(self) -> None:
        """
        Tests that pushing through a not-yet-materialized default attribute
        works and does not alias the schema default.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        app.execute_data(data_text=(
            'u = T(name = "idle");\n'
            'u.jobs.push("later");\n'
        ))

        instance: Any = app.query(struct_name="T").first()

        assert list(instance.get_attribute(name="jobs").get()) == ["later"]

        # A fresh instance must not inherit the pushed element
        app.execute_data(data_text='v = T(name = "fresh");\n')

        fresh: Any = app.query(struct_name="T").to_list()[1]

        assert list(fresh.get_attribute(name="jobs").get()) == []


class TestQueuePersistence:
    """
    Tests queues across the v2 binary persistence format.
    """

    def _round_trip(self, data_text: str) -> tuple[Any, Any]:
        """
        Dumps and reloads a database twice.

        Args:
            data_text (str): .nadata source.

        Returns:
            tuple[Any, Any]: First and second round trip instance mappings.
        """

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(content=data_text)

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "q.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        storage_path2: str = os.path.join(temp_dir, "q2.ndb")

        dump_database(database=reloaded, storage_path=storage_path2)

        reloaded2: Any = load_database(storage_path=storage_path2)

        return (
            reloaded.tables_by_struct_name["T"].instances,
            reloaded2.tables_by_struct_name["T"].instances,
        )

    def test_round_trip(self) -> None:
        """
        Tests both queue kinds across two dump/load cycles.
        """

        lazy: Any
        lazy2: Any

        lazy, lazy2 = self._round_trip(data_text=(
            't = T(name = "w", jobs = ["j1", "j2"]);\n'
            't.pq.push("cleanup", 5);\n'
            't.pq.push("fire", 1);\n'
            't.pqf.push("x", 1.5);\n'
        ))

        first: Any = lazy[0]

        assert list(first.get_attribute(name="jobs").get()) == ["j1", "j2"]
        assert first.get_attribute(name="pq").get() == [(1, "fire"), (5, "cleanup")]
        assert first.get_attribute(name="pqf").get() == [(1.5, "x")]

        second: Any = lazy2[0]

        assert second.get_attribute(name="pq").get() == [(1, "fire"), (5, "cleanup")]

    def test_lazy_open_undecoded(self) -> None:
        """
        Tests that queue columns stay pending on lazy open.
        """

        from nadata.core.database.persistence.lazy_instances import LazyInstanceDict

        app: Nadata = Nadata(database=NaDataDatabase())

        app.execute_schema(schema_text=SCHEMA)

        NaDataLoader(database=app.database).load_file(
            content='t = T(name = "w", jobs = ["j1"]);\n'
        )

        temp_dir: str = tempfile.mkdtemp()
        storage_path: str = os.path.join(temp_dir, "lazy.ndb")

        dump_database(database=app.database, storage_path=storage_path)

        reloaded: Any = load_database(storage_path=storage_path)

        lazy: Any = reloaded.tables_by_struct_name["T"].instances

        assert isinstance(lazy, LazyInstanceDict)
        assert dict.keys(lazy) == dict.keys({})
        assert len(lazy) == 1
