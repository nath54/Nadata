"""
Tests for the QuerySet fluent API: ordering, existence checks, plucking,
and numeric aggregation.
"""

# Import Modules
from __future__ import annotations
from typing import Any

import pytest

from src.api import Nadata
from src.core.database.database import NaDataDatabase
from src.parser.nadata_loader.loader import NaDataLoader

SCHEMA: str = """
class Person {
    name: Str;
    age: Int32;
    city: Str = "Paris";
    tags: Array[Str] = [];
}
"""

DATA: str = """
ann = Person(name = "ann", age = 40);
bob = Person(name = "bob", age = 20);
cid = Person(name = "cid", age = 35);
dee = Person(name = "dee", age = 20);
"""


def _app() -> Nadata:
    """
    Builds an app with the Person schema and seeded data.

    Returns:
        Nadata: The prepared application facade.
    """

    app: Nadata = Nadata(database=NaDataDatabase())

    app.execute_schema(schema_text=SCHEMA)
    NaDataLoader(database=app.database).load_file(content=DATA)

    return app


def _names(queryset: Any) -> list[str]:
    """
    Extracts the name payloads of a queryset in order.

    Args:
        queryset (Any): QuerySet to read.

    Returns:
        list[str]: Name payloads.
    """

    return [
        person.attributes["name"].get()
        for person in queryset.to_list()
    ]


class TestOrdering:
    """
    Tests order_by behavior.
    """

    def test_ascending(self) -> None:
        """
        Tests ascending sort by age with a stable order for ties.
        """

        app: Nadata = _app()

        assert _names(app.query(struct_name="Person").order_by(attribute="age")) == [
            "bob", "dee", "cid", "ann",
        ]

    def test_descending(self) -> None:
        """
        Tests descending sort by age.
        """

        app: Nadata = _app()

        assert _names(
            app.query(struct_name="Person").order_by(attribute="age", descending=True)
        ) == ["ann", "cid", "bob", "dee"]

    def test_chained_with_limit(self) -> None:
        """
        Tests ordering chained with limit.
        """

        app: Nadata = _app()

        assert _names(
            app.query(struct_name="Person")
            .order_by(attribute="age", descending=True)
            .limit(count=2)
        ) == ["ann", "cid"]

    def test_unknown_attribute_rejected(self) -> None:
        """
        Tests that ordering by an undeclared attribute raises AttributeError.
        """

        app: Nadata = _app()

        with pytest.raises(AttributeError):
            app.query(struct_name="Person").order_by(attribute="nope")


class TestExistenceAndPluck:
    """
    Tests exists() and pluck().
    """

    def test_exists_true_and_false(self) -> None:
        """
        Tests existence checks on matching and empty sets.
        """

        app: Nadata = _app()

        assert app.query(struct_name="Person").exists()

        assert not app.query(struct_name="Person").where(
            predicate=lambda person: False
        ).exists()

    def test_pluck(self) -> None:
        """
        Tests extracting raw attribute payloads.
        """

        app: Nadata = _app()

        assert app.query(struct_name="Person").pluck(attribute="name") == [
            "ann", "bob", "cid", "dee",
        ]

    def test_pluck_lazy_default(self) -> None:
        """
        Tests plucking an attribute that only exists as a lazy default.
        """

        app: Nadata = _app()

        assert app.query(struct_name="Person").pluck(attribute="city") == [
            "Paris", "Paris", "Paris", "Paris",
        ]


class TestAggregation:
    """
    Tests numeric aggregation helpers.
    """

    def test_sum_min_max_average(self) -> None:
        """
        Tests the four numeric aggregations.
        """

        app: Nadata = _app()
        queryset: Any = app.query(struct_name="Person")

        assert queryset.sum_of(attribute="age") == 115
        assert queryset.min_of(attribute="age") == 20
        assert queryset.max_of(attribute="age") == 40
        assert queryset.average_of(attribute="age") == 28.75

    def test_filtered_aggregation(self) -> None:
        """
        Tests aggregations over a filtered set.
        """

        app: Nadata = _app()

        queryset: Any = app.query(struct_name="Person").where(
            predicate=lambda person: person.attributes["age"].get() > 20
        )

        assert queryset.sum_of(attribute="age") == 75

    def test_non_numeric_rejected(self) -> None:
        """
        Tests that aggregating a non-numeric attribute raises TypeError.
        """

        app: Nadata = _app()

        with pytest.raises(TypeError):
            app.query(struct_name="Person").sum_of(attribute="name")


class TestLanguageSide:
    """
    Tests queryset helpers invoked from data scripts.
    """

    def test_language_order_and_aggregate(self) -> None:
        """
        Tests order_by / sum_of / exists through the data loader.
        """

        app: Nadata = _app()

        NaDataLoader(database=app.database).load_file(
            content="""
oldest = Person.order_by("age", descending = true).first();
print(oldest.name);
print(Person.sum_of("age"), Person.average_of("age"));
print(Person.where((p: Person) -> p.age > 30).exists());
for(p: Person in Person.order_by("age").limit(2)){ print(p.name); }
"""
        )
