"""
Tests for filtered random sampling: StructTable.sample / sample_by,
the struct binding's sample / sample_by / sample_where entry points,
the Nadata facade sample(), and QuerySet.sample().

Sampling draws random rows without decoding the whole table, probing
until enough match and falling back to a scan for sparse filters.
"""

# Import Modules
import random
import tempfile
from pathlib import Path
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


SCHEMA: str = """
class Item {
    prompt: Str;
    kind: Str = "qcm";
    difficulty: Int32 = 1;
}
class Special(Item) {
    bonus: Str = "";
}
"""


def _populated_app(item_count: int = 5_000) -> Any:
    """
    Creates an app with a schema and a polymorphic item population:
    every 10th item is a Special. kind is 'written' for every third
    item, 'qcm' otherwise; difficulty cycles 0..4.

    Args:
        item_count (int): Number of Item rows.

    Returns:
        Any: The populated Nadata instance.
    """

    random.seed(1234)

    app: Any = _app()
    tmp: Path = Path(tempfile.mkdtemp())
    project: Path = tmp / "p"
    (project / "structs").mkdir(parents=True)
    (project / "data").mkdir(parents=True)
    (project / "structs" / "schema.nastruct").write_text(SCHEMA)
    (project / "data" / "seed.nadata").write_text("#init\nflush();\n")

    loaded: Any = Nadata.load(path=str(project))

    lines: list[str] = []

    for i in range(item_count):
        kind: str = "written" if i % 3 == 0 else "qcm"
        lines.append(
            f'it{i} = Item(nid = "i:{i}", prompt = "p{i}", '
            f'kind = "{kind}", difficulty = {i % 5});'
        )

        if i % 10 == 0:
            lines.append(
                f's{i} = Special(nid = "s:{i}", prompt = "sp{i}", '
                f'kind = "qcm", difficulty = 0, bonus = "b");'
            )

    loaded.execute_data("\n".join(lines) + "\nflush();")

    return loaded


def _binding(app: Any, struct_name: str) -> Any:
    """
    Returns the language runtime binding of a struct.

    Args:
        app (Any): The Nadata facade.
        struct_name (str): Struct name.

    Returns:
        Any: The StructRuntimeBinding.
    """

    app._ensure_globals()

    return app._globals[struct_name]


class TestStructTableSample:
    """
    Core random sampling over a storage table.
    """

    def test_unfiltered_sample_count_and_distinctness(self) -> None:
        """
        sample() returns exactly count distinct instances.
        """

        app: Any = _populated_app()
        table: Any = app.database.tables_by_struct_name["Item"]

        drawn: list[Any] = table.sample(count=50)

        assert len(drawn) == 50
        assert len({inst.entity_id.value for inst in drawn}) == 50

    def test_sample_is_random(self) -> None:
        """
        Two consecutive draws differ (random ordering).
        """

        app: Any = _populated_app()
        table: Any = app.database.tables_by_struct_name["Item"]

        first: list[Any] = table.sample(count=10)
        second: list[Any] = table.sample(count=10)

        assert [inst.entity_id.value for inst in first] != [
            inst.entity_id.value for inst in second
        ]

    def test_sample_is_seedable(self) -> None:
        """
        A seeded rng reproduces the same draw.
        """

        app: Any = _populated_app()
        table: Any = app.database.tables_by_struct_name["Item"]

        first: list[Any] = table.sample(count=10, rng=random.Random(42))
        second: list[Any] = table.sample(count=10, rng=random.Random(42))

        assert [inst.entity_id.value for inst in first] == [
            inst.entity_id.value for inst in second
        ]

    def test_sample_without_full_decode(self) -> None:
        """
        Sampling k rows from a freshly reloaded lazy table decodes
        O(k) rows, not the whole table.
        """

        app: Any = _populated_app(item_count=20_000)
        store: Path = Path(tempfile.mkdtemp()) / "s.ndb"
        app.dump(str(store))

        reloaded: Any = Nadata.open(str(store))
        table: Any = reloaded.database.tables_by_struct_name["Item"]

        if not hasattr(table.instances, "all_local_ids"):
            pytest.skip("in-memory table (no lazy rows)")

        table.sample(count=25)

        decoded: int = len(dict.keys(table.instances))

        assert decoded < 500

    def test_predicated_sample_rows_all_match(self) -> None:
        """
        The predicate probe path returns only matching rows.
        """

        app: Any = _populated_app()
        table: Any = app.database.tables_by_struct_name["Item"]

        drawn: list[Any] = table.sample(
            count=20,
            predicate=lambda inst: inst.attributes["kind"].get() == "written",
        )

        assert len(drawn) == 20
        assert all(
            inst.attributes["kind"].get() == "written" for inst in drawn
        )

    def test_sparse_predicate_falls_back_to_scan(self) -> None:
        """
        A filter matching only a handful of rows still fills the quota
        through the fallback scan.
        """

        app: Any = _populated_app()
        table: Any = app.database.tables_by_struct_name["Item"]

        # difficulty == 4 matches 1/5 of rows; probe_factor=1 forces
        # the fallback path for a large enough count
        drawn: list[Any] = table.sample(
            count=200,
            predicate=lambda inst: inst.attributes["difficulty"].get() == 4,
            probe_factor=1,
        )

        assert len(drawn) == 200
        assert all(
            inst.attributes["difficulty"].get() == 4 for inst in drawn
        )

    def test_sample_more_than_available(self) -> None:
        """
        Asking for more rows than exist returns every row.
        """

        app: Any = _populated_app()
        table: Any = app.database.tables_by_struct_name["Special"]

        if hasattr(table.instances, "all_local_ids"):
            expected: int = len(table.instances.all_local_ids())
        else:
            expected: int = len(table.instances)

        drawn: list[Any] = table.sample(count=10_000)

        assert len(drawn) == expected

    def test_sample_empty_table(self) -> None:
        """
        Sampling an empty table returns an empty list.
        """

        tmp: Path = Path(tempfile.mkdtemp())
        project: Path = tmp / "p"
        (project / "structs").mkdir(parents=True)
        (project / "data").mkdir(parents=True)
        (project / "structs" / "schema.nastruct").write_text(
            "class Solo {\n    name: Str = \"\";\n}\n"
        )
        (project / "data" / "seed.nadata").write_text("#init\nflush();\n")

        app: Any = Nadata.load(path=str(project))
        table: Any = app.database.tables_by_struct_name["Solo"]

        assert table.sample(count=5) == []


class TestStructTableSampleBy:
    """
    Indexed equality sampling.
    """

    def test_indexed_sample_by(self) -> None:
        """
        sample_by draws only matching rows through the index.
        """

        app: Any = _populated_app()
        app.execute_data('index(Item, "kind");')

        table: Any = app.database.tables_by_struct_name["Item"]

        drawn: list[Any] = table.sample_by(
            attribute="kind", raw="written", count=30
        )

        assert len(drawn) == 30
        assert all(
            inst.attributes["kind"].get() == "written" for inst in drawn
        )

    def test_sample_by_without_index_returns_none(self) -> None:
        """
        No index on the attribute signals the caller to fall back.
        """

        app: Any = _populated_app()
        table: Any = app.database.tables_by_struct_name["Item"]

        assert table.sample_by(attribute="prompt", raw="p1", count=3) is None

    def test_sample_by_missing_bucket(self) -> None:
        """
        An indexed value with no rows returns an empty list.
        """

        app: Any = _populated_app()
        app.execute_data('index(Item, "kind");')

        table: Any = app.database.tables_by_struct_name["Item"]

        assert (
            table.sample_by(attribute="kind", raw="nope", count=3) == []
        )


class TestBindingSampling:
    """
    Language-visible sampling on struct bindings (polymorphic).
    """

    def test_binding_sample_includes_subclasses(self) -> None:
        """
        Item.sample() draws from Item and Special tables.
        """

        app: Any = _populated_app()
        binding: Any = _binding(app=app, struct_name="Item")

        drawn: list[Any] = binding.sample(count=200)

        assert len(drawn) == 200

        names: set[str] = {inst.struct.name for inst in drawn}

        assert names == {"Item", "Special"}

    def test_binding_sample_by(self) -> None:
        """
        sample_by routes through the index and falls back without one.
        """

        app: Any = _populated_app()
        app.execute_data('index(Item, "kind");')
        binding: Any = _binding(app=app, struct_name="Item")

        drawn: list[Any] = binding.sample_by(
            attribute="kind", raw="written", count=25
        )

        assert len(drawn) == 25
        assert all(
            inst.attributes["kind"].get() == "written" for inst in drawn
        )

        # unindexed attribute: probe fallback path
        drawn: list[Any] = binding.sample_by(
            attribute="prompt", raw="p7", count=5
        )

        assert len(drawn) == 1
        assert drawn[0].attributes["prompt"].get() == "p7"

    def test_binding_sample_where(self) -> None:
        """
        sample_where probes with a lambda predicate.
        """

        app: Any = _populated_app()
        binding: Any = _binding(app=app, struct_name="Item")

        drawn: list[Any] = binding.sample_where(
            predicate=lambda inst: inst.attributes["difficulty"].get() == 0,
            count=15,
        )

        assert len(drawn) == 15
        assert all(
            inst.attributes["difficulty"].get() == 0 for inst in drawn
        )

    def test_binding_sample_where_rejects_non_callable(self) -> None:
        """
        A non-callable predicate raises TypeError.
        """

        app: Any = _populated_app()
        binding: Any = _binding(app=app, struct_name="Item")

        with pytest.raises(TypeError):
            binding.sample_where(predicate="nope", count=5)

    def test_language_script_sampling(self) -> None:
        """
        Sampling runs from .nadata source, including lambda syntax;
        results surface through probe instances.
        """

        tmp: Path = Path(tempfile.mkdtemp())
        project: Path = tmp / "p"
        (project / "structs").mkdir(parents=True)
        (project / "data").mkdir(parents=True)
        (project / "structs" / "schema.nastruct").write_text(
            SCHEMA + "\nclass Probe {\n    n: Int32 = 0;\n}\n"
        )
        (project / "data" / "seed.nadata").write_text("#init\nflush();\n")

        app: Any = Nadata.load(path=str(project))

        lines: list[str] = []

        for i in range(5_000):
            kind: str = "written" if i % 3 == 0 else "qcm"
            lines.append(
                f'it{i} = Item(nid = "i:{i}", prompt = "p{i}", '
                f'kind = "{kind}", difficulty = {i % 5});'
            )

        app.execute_data("\n".join(lines) + "\nflush();\n")
        app.execute_data('index(Item, "kind");')

        app.execute_data(
            'n1: Int32 = 0;\n'
            'for (it: Item in Item.sample(30)) { n1 = n1 + 1; }\n'
            'n2: Int32 = 0;\n'
            'for (it: Item in Item.sample_by("kind", "written", 20)) '
            '{ n2 = n2 + 1; }\n'
            'n3: Int32 = 0;\n'
            'for (it: Item in '
            'Item.sample_where((it2: Item) -> it2.difficulty == 0, 15)) '
            '{ n3 = n3 + 1; }\n'
            # Constructor args evaluate before script statements, so the
            # counts land through member assignment (script order).
            'p1 = Probe(nid = "p:draw");\n'
            'p1.n = n1;\n'
            'p2 = Probe(nid = "p:tagged");\n'
            'p2.n = n2;\n'
            'p3 = Probe(nid = "p:easy");\n'
            'p3.n = n3;\n'
        )

        probe: Any = app.database.get_by_nid

        assert probe(struct_name="Probe", nid="p:draw").attributes["n"].get() == 30
        assert probe(struct_name="Probe", nid="p:tagged").attributes["n"].get() == 20
        assert probe(struct_name="Probe", nid="p:easy").attributes["n"].get() == 15


class TestFacadeSample:
    """
    Nadata.sample() over the whole hierarchy.
    """

    def test_facade_sample_unfiltered(self) -> None:
        """
        Unfiltered facade sampling returns distinct instances.
        """

        app: Any = _populated_app()

        drawn: list[Any] = app.sample(struct_name="Item", count=50)

        assert len(drawn) == 50
        assert len({inst.entity_id.value for inst in drawn}) == 50

    def test_facade_sample_with_predicate(self) -> None:
        """
        Predicated facade sampling returns only matching rows.
        """

        app: Any = _populated_app()

        drawn: list[Any] = app.sample(
            struct_name="Item",
            count=10,
            predicate=lambda inst: inst.struct.name == "Special",
        )

        assert len(drawn) == 10
        assert all(inst.struct.name == "Special" for inst in drawn)

    def test_facade_sample_unknown_struct(self) -> None:
        """
        An unknown struct name raises KeyError.
        """

        app: Any = _populated_app()

        with pytest.raises(KeyError):
            app.sample(struct_name="Nope", count=5)

    def test_facade_sample_empty_result(self) -> None:
        """
        Count <= 0 returns an empty list without touching tables.
        """

        app: Any = _populated_app()

        assert app.sample(struct_name="Item", count=0) == []


class TestQuerySetSample:
    """
    Sampling an already filtered candidate set.
    """

    def test_queryset_sample(self) -> None:
        """
        sample() draws from the filtered candidates only.
        """

        app: Any = _populated_app()

        queryset: Any = app.query(struct_name="Item").where(
            predicate=lambda inst: inst.attributes["kind"].get() == "written"
        )

        drawn: list[Any] = queryset.sample(count=7)

        assert len(drawn) == 7
        assert all(
            inst.attributes["kind"].get() == "written" for inst in drawn
        )

    def test_queryset_sample_over_count(self) -> None:
        """
        Sampling more than the candidate count returns all candidates.
        """

        app: Any = _populated_app()

        queryset: Any = app.query(struct_name="Special")

        expected: int = queryset.count()

        drawn: list[Any] = queryset.sample(count=10_000)

        assert len(drawn) == expected

    def test_queryset_sample_seeded(self) -> None:
        """
        A seeded rng reproduces the same draw.
        """

        app: Any = _populated_app()

        queryset: Any = app.query(struct_name="Item")

        first: list[Any] = queryset.sample(count=10, rng=random.Random(7))
        second: list[Any] = queryset.sample(count=10, rng=random.Random(7))

        assert [inst.entity_id.value for inst in first] == [
            inst.entity_id.value for inst in second
        ]
