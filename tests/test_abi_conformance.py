"""
ABI conformance suite — the canonical definition of native ABI behavior.

Every test carries a stable case id (`@abi_case("nd_x/scenario")`)
mirroring a `native/nadata.h` function. The engine behaviors asserted
here are the contract: bindings in Kotlin, Java, Rust, JavaScript, ...
replay the same case ids against their own wrappers, so per-binding
coverage is diffable against this file.

Status expectations come from `tests/abi/contract.py` (NdStatus);
scenarios come from `tests/abi/fixtures.py`. Both keep this file free
of literals that could drift.
"""

# Import Modules
from typing import Any

import pytest

from abi.contract import (
    HOME_TYPES_NOT_REPRESENTABLE,
    NdStatus,
    abi_case,
    registered_case_ids,
    status_for,
)
from abi.fixtures import (
    EX0_ANIMALS,
    EX0_ENTITIES,
    EX0_HUMANS,
    EX0_TOTAL,
    copied_ex0,
    dumped_ex0,
    loaded_ex0,
    numeric_project,
    session_project,
    where_str,
)
from nadata.api import Nadata
from nadata.core.types.home_types.datetime import DateTime


# ------------------------------------------------------------------ #
# Lifecycle                                                           #
# ------------------------------------------------------------------ #


class TestLifecycle:
    """nd_load / nd_open / nd_open_folder contracts."""

    @abi_case("nd_load/project_folder")
    def test_load_project_folder(self) -> None:
        """A folder loads schemas then data, exposing symbols."""

        app: Nadata = loaded_ex0()

        assert "Human" in app.struct_names
        assert "tom" in app.symbol_names
        assert "is_adult" in app.function_names

    @abi_case("nd_load/schema_file")
    def test_load_single_schema_file(self) -> None:
        """A lone .nastruct file loads with no data symbols."""

        app: Nadata = Nadata.load(
            path="./examples/ex0_simple/structs/structs.nastruct"
        )

        assert "Human" in app.struct_names
        assert app.symbol_names == []

    @abi_case("nd_load/data_file_alone")
    def test_load_data_file_alone_is_value_error(self) -> None:
        """A .nadata file without its schema -> ND_ERR_VALUE."""

        with pytest.raises(ValueError) as info:
            Nadata.load(path="./examples/ex0_simple/data/data1.nadata")

        assert status_for(info.value) is NdStatus.ND_ERR_VALUE

    @abi_case("nd_load/unsupported_path")
    def test_load_unsupported_path_is_value_error(self) -> None:
        """An unknown extension / missing path -> ND_ERR_VALUE."""

        with pytest.raises(ValueError) as info:
            Nadata.load(path="./examples/does_not_exist.txt")

        assert status_for(info.value) is NdStatus.ND_ERR_VALUE

    @abi_case("nd_open/roundtrip")
    def test_open_roundtrip(self, tmp_path: Any) -> None:
        """An .ndb dumped from a project reopens with equal counts."""

        storage_path: str = dumped_ex0(tmp_path)

        reopened: Nadata = Nadata.open(storage_path=storage_path)

        assert reopened.count("Human") == EX0_HUMANS
        assert reopened.total_instance_count() == EX0_TOTAL

    @abi_case("nd_open/missing_file")
    def test_open_missing_file_is_value_error(self) -> None:
        """Opening a non-existent .ndb -> ND_ERR_VALUE (dump first)."""

        with pytest.raises(ValueError) as info:
            Nadata.open(storage_path="./definitely_missing_store.ndb")

        assert status_for(info.value) is NdStatus.ND_ERR_VALUE

    @abi_case("nd_open/invalid_database")
    def test_open_invalid_database_is_value_error(self) -> None:
        """Opening a garbage file -> ND_ERR_VALUE."""

        garbage: Any = tmp_garbage()

        with pytest.raises(ValueError) as info:
            Nadata.open(storage_path=garbage)

        assert status_for(info.value) is NdStatus.ND_ERR_VALUE

    @abi_case("nd_open_folder/session_pipeline")
    def test_open_folder_session(self, tmp_path: Any) -> None:
        """A session folder opens and creates its world .ndb."""

        project: Any = copied_ex0(tmp_path)

        app: Nadata = Nadata.open_folder(folder_path=str(project))

        assert "Human" in app.struct_names
        assert (project / "nadata.ndb").exists()


def tmp_garbage() -> str:
    """
    Creates a non-Nadata file for the invalid-database case.

    Returns:
        str: Path to the garbage file.
    """

    import tempfile

    handle, path = tempfile.mkstemp(suffix=".ndb")

    with open(handle, "w", encoding="utf-8") as stream:
        stream.write("this is not a database")

    return path


# ------------------------------------------------------------------ #
# Introspection                                                       #
# ------------------------------------------------------------------ #


class TestIntrospection:
    """nd_struct_names / nd_count / nd_summary contracts."""

    @abi_case("nd_struct_names/ex0")
    def test_struct_names(self) -> None:
        """All schema structs are listed."""

        names: list[str] = loaded_ex0().struct_names

        for expected in ("Entity", "Human", "Animal", "Cat"):
            assert expected in names

    @abi_case("nd_function_names/ex0")
    def test_function_names(self) -> None:
        """Standalone schema functions are listed."""

        assert "is_adult" in loaded_ex0().function_names

    @abi_case("nd_symbol_names/ex0")
    def test_symbol_names(self) -> None:
        """Bound data symbols are listed."""

        names: list[str] = loaded_ex0().symbol_names

        for expected in ("tom", "jean", "louise", "white_cat"):
            assert expected in names

    @abi_case("nd_summary/nonempty_string")
    def test_summary_is_string(self) -> None:
        """summary() returns a non-empty human-readable string."""

        summary: str = loaded_ex0().summary()

        assert isinstance(summary, str) and len(summary) > 0

    @abi_case("nd_count/polymorphic")
    def test_count_is_polymorphic(self) -> None:
        """Counts include every subclass (Animal counts the Cat)."""

        app: Nadata = loaded_ex0()

        assert app.count("Human") == EX0_HUMANS
        assert app.count("Animal") == EX0_ANIMALS
        assert app.count("Entity") == EX0_ENTITIES

    @abi_case("nd_count/unknown_struct")
    def test_count_unknown_struct(self) -> None:
        """Unknown struct -> ND_ERR_NOTFOUND (KeyError)."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.count("NoSuch")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_total_instance_count/ex0")
    def test_total_instance_count(self) -> None:
        """The total spans every table."""

        assert loaded_ex0().total_instance_count() == EX0_TOTAL


# ------------------------------------------------------------------ #
# Instances and attributes                                            #
# ------------------------------------------------------------------ #


class TestInstances:
    """nd_get_symbol_id / nd_get_attribute / nd_create_instance / calls."""

    @abi_case("nd_get_symbol_id/ok")
    def test_get_symbol_id(self) -> None:
        """A bound symbol resolves to a stable instance."""

        app: Nadata = loaded_ex0()
        tom: Any = app.get(symbol="tom")

        assert app.get(symbol="tom") is tom

    @abi_case("nd_get_symbol_id/unknown_symbol")
    def test_get_symbol_id_unknown(self) -> None:
        """Unknown symbol -> ND_ERR_NOTFOUND (KeyError)."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.get(symbol="nobody")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_get_attribute/plain_scalar")
    def test_get_attribute_plain_scalar(self) -> None:
        """Attributes unwrap to plain Python values."""

        app: Nadata = loaded_ex0()

        assert app.get_attribute(symbol="tom", name="name") == "Tom"

    @abi_case("nd_get_attribute/unknown_symbol")
    def test_get_attribute_unknown_symbol(self) -> None:
        """Unknown symbol -> ND_ERR_NAME."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.get_attribute(symbol="nobody", name="name")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_get_attribute/unknown_attribute")
    def test_get_attribute_unknown_attribute(self) -> None:
        """Unknown attribute -> ND_ERR_NOTFOUND (engine AttributeError)."""

        app: Nadata = loaded_ex0()

        with pytest.raises(AttributeError) as info:
            app.get_attribute(symbol="tom", name="no_such")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_get_attribute/home_type_payload")
    def test_get_attribute_home_type(self) -> None:
        """
        Home-type payloads (DateTime) are NOT plain values: the shim
        maps them to ND_ERR_TYPE (see HOME_TYPES_NOT_REPRESENTABLE).
        This test pins the engine behavior the policy is based on.
        """

        birthday: Any = loaded_ex0().get_attribute(
            symbol="tom", name="birthday"
        )

        assert isinstance(birthday, DateTime)

    @abi_case("nd_set_attribute/roundtrip")
    def test_set_attribute_roundtrip(self) -> None:
        """A set value reads back identically."""

        app: Nadata = loaded_ex0()

        app.set_attribute(symbol="tom", name="birthplace", value="Lyon")

        assert app.get_attribute(symbol="tom", name="birthplace") == "Lyon"

    @abi_case("nd_set_attribute/type_mismatch")
    def test_set_attribute_type_mismatch(self) -> None:
        """A wrong-typed value -> ND_ERR_TYPE."""

        app: Nadata = loaded_ex0()

        with pytest.raises(TypeError) as info:
            app.set_attribute(symbol="tom", name="name", value=123)

        assert status_for(info.value) is NdStatus.ND_ERR_TYPE

    @abi_case("nd_set_attribute/unknown_symbol")
    def test_set_attribute_unknown_symbol(self) -> None:
        """Unknown symbol -> ND_ERR_NAME."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.set_attribute(symbol="nobody", name="name", value="x")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_create_instance/with_attributes_and_symbol")
    def test_create_instance_with_symbol(self) -> None:
        """Created instances are bound and visible to later calls."""

        app: Nadata = loaded_ex0()
        before: int = app.count("Human")

        app.create_instance(
            struct_name="Human",
            attributes={"name": "Zoe", "birthplace": "Lyon"},
            symbol="zoe",
        )

        assert app.count("Human") == before + 1
        assert app.get_attribute(symbol="zoe", name="name") == "Zoe"

    @abi_case("nd_create_instance/defaults_applied")
    def test_create_instance_defaults(self) -> None:
        """Omitted attributes take their schema defaults."""

        app: Nadata = loaded_ex0()

        app.create_instance(
            struct_name="ObjectType", attributes={"name": "Sword"}
        )

        # symbolless create binds nothing; assert via a query
        matches: Any = app.query("ObjectType").where(
            lambda inst: inst.get_attribute("name").get() == "Sword"
        )

        assert matches.count() == 1
        assert matches.instances[0].get_attribute("description").get() == ""

    @abi_case("nd_create_instance/unknown_struct")
    def test_create_instance_unknown_struct(self) -> None:
        """Unknown struct -> ND_ERR_NOTFOUND."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.create_instance(struct_name="NoSuch", attributes={})

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_call_function/positional_args")
    def test_call_function_positional(self) -> None:
        """Standalone functions accept plain Python values."""

        app: Nadata = loaded_ex0()
        tom: Any = app.get(symbol="tom")

        assert app.call_function("is_adult", tom) is True

    @abi_case("nd_call_function/unknown_function")
    def test_call_function_unknown(self) -> None:
        """Unknown function -> ND_ERR_NOTFOUND."""

        with pytest.raises(KeyError) as info:
            loaded_ex0().call_function("no_such_function")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_call_function/argument_type_mismatch")
    def test_call_function_argument_mismatch(self) -> None:
        """A wrong-typed argument -> ND_ERR_TYPE."""

        app: Nadata = loaded_ex0()

        with pytest.raises(TypeError) as info:
            app.call_function("is_adult", 42)

        assert status_for(info.value) is NdStatus.ND_ERR_TYPE

    @abi_case("nd_call_function/home_type_result")
    def test_call_function_home_type_result(self) -> None:
        """
        A function returning a home type (DateTime) pins the v1 ABI
        limitation: the shim maps such results to ND_ERR_TYPE. See
        HOME_TYPES_NOT_REPRESENTABLE.
        """

        app: Nadata = loaded_ex0()
        tom: Any = app.get(symbol="tom")

        difference: Any = app.call_function(
            "calculate_age_difference", h1=tom, h2=tom
        )

        assert isinstance(difference, DateTime)

    @abi_case("nd_call_method/ok")
    def test_call_method(self) -> None:
        """Methods dispatch on the symbol's instance."""

        app: Nadata = loaded_ex0()
        tom: Any = app.get(symbol="tom")
        jean: Any = app.get(symbol="jean")

        assert app.call_method("tom", "older", jean) is False
        assert app.call_method("jean", "older", tom) is True

    @abi_case("nd_call_method/unknown_method")
    def test_call_method_unknown(self) -> None:
        """Unknown method -> ND_ERR_NOTFOUND."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.call_method("tom", "no_such_method")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_call_method/unknown_symbol")
    def test_call_method_unknown_symbol(self) -> None:
        """Unknown receiver symbol -> ND_ERR_NAME."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.call_method("nobody", "older")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND


# ------------------------------------------------------------------ #
# Queries                                                             #
# ------------------------------------------------------------------ #


class TestQueries:
    """nd_query_* contracts. Order is instance order (stable)."""

    @abi_case("nd_query_new/unknown_struct")
    def test_query_unknown_struct(self) -> None:
        """Unknown struct -> ND_ERR_NOTFOUND."""

        app: Nadata = loaded_ex0()

        with pytest.raises(KeyError) as info:
            app.query("NoSuch")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_query_where_str/engine_evaluated")
    def test_where_str(self) -> None:
        """String predicates evaluate engine-side (the universal path)."""

        app: Nadata = loaded_ex0()

        matches: Any = where_str(app.query("Human"), "h", 'h.name == "Tom"')

        assert matches.pluck("name") == ["Tom"]

    @abi_case("nd_query_where_str/syntax_error")
    def test_where_str_syntax_error(self) -> None:
        """A malformed predicate text -> ND_ERR_SYNTAX."""

        app: Nadata = loaded_ex0()

        with pytest.raises(SyntaxError) as info:
            where_str(app.query("Human"), "h", "h.name ====")

        assert status_for(info.value) is NdStatus.ND_ERR_SYNTAX

    @abi_case("nd_query_where_str/type_error")
    def test_where_str_type_error(self) -> None:
        """A type-invalid predicate -> ND_ERR_TYPE during evaluation."""

        app: Nadata = loaded_ex0()

        with pytest.raises(TypeError) as info:
            where_str(app.query("Human"), "h", 'h.name < 5')

        assert status_for(info.value) is NdStatus.ND_ERR_TYPE

    @abi_case("nd_query_where_str/empty_result")
    def test_where_str_empty(self) -> None:
        """Non-matching predicates yield an empty set, not an error."""

        app: Nadata = loaded_ex0()

        matches: Any = where_str(
            app.query("Human"), "h", 'h.name == "Nobody"'
        )

        assert matches.count() == 0
        assert matches.exists() is False

    @abi_case("nd_query_where_cb/preserves_order")
    def test_where_callback_preserves_order(self) -> None:
        """Callback filters keep instance order (chunk-parallel safe)."""

        app: Nadata = loaded_ex0()

        parisians: Any = app.query("Human").where(
            lambda inst: inst.get_attribute("birthplace").get() == "Paris"
        )

        assert parisians.pluck("name") == ["Tom", "Jean", "Louise"]

    @abi_case("nd_query_order_by/ascending_and_descending")
    def test_order_by(self) -> None:
        """order_by sorts stably both ways on raw payloads."""

        app: Nadata = loaded_ex0()

        ascending: Any = app.query("Human").order_by("name")
        descending: Any = app.query("Human").order_by("name", descending=True)

        assert ascending.pluck("name") == ["Jean", "Louise", "Tom"]
        assert descending.pluck("name") == ["Tom", "Louise", "Jean"]

    @abi_case("nd_query_order_by/unknown_attribute")
    def test_order_by_unknown_attribute(self) -> None:
        """Schema-validated attribute -> ND_ERR_NOTFOUND when unknown."""

        app: Nadata = loaded_ex0()

        with pytest.raises(AttributeError) as info:
            app.query("Human").order_by("no_such")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_query_limit/window")
    def test_limit_window(self) -> None:
        """limit() takes the first n after ordering."""

        app: Nadata = loaded_ex0()

        window: Any = app.query("Human").order_by("name").limit(2)

        assert window.pluck("name") == ["Jean", "Louise"]

    @abi_case("nd_query_count/after_chaining")
    def test_count_after_chaining(self) -> None:
        """count reflects every chained filter."""

        app: Nadata = loaded_ex0()

        chained: Any = app.query("Human").order_by("name").limit(2)

        assert chained.count() == 2

    @abi_case("nd_query_exists/true_false")
    def test_exists(self) -> None:
        """exists() reflects non-emptiness."""

        app: Nadata = loaded_ex0()

        assert app.query("Human").exists() is True
        assert where_str(
            app.query("Human"), "h", 'h.name == "Nobody"'
        ).exists() is False

    @abi_case("nd_query_first_id/empty_set_is_notfound")
    def test_first_empty_is_notfound(self) -> None:
        """
        first() on an empty set is None in Python; the ABI contract is
        ND_ERR_NOTFOUND (explicit, catchable) — ids are 0-based, so no
        sentinel value would be safe.
        """

        empty: Any = where_str(
            loaded_ex0().query("Human"), "h", 'h.name == "Nobody"'
        )

        assert empty.first() is None
        assert empty.last() is None

    @abi_case("nd_query_pluck/order_and_payloads")
    def test_pluck(self) -> None:
        """pluck returns raw payloads in instance order."""

        assert loaded_ex0().query("Human").pluck("name") == [
            "Tom", "Jean", "Louise",
        ]

    @abi_case("nd_query_pluck/unknown_attribute")
    def test_pluck_unknown_attribute(self) -> None:
        """Unknown attribute -> ND_ERR_NOTFOUND."""

        app: Nadata = loaded_ex0()

        with pytest.raises(AttributeError) as info:
            app.query("Human").pluck("no_such")

        assert status_for(info.value) is NdStatus.ND_ERR_NOTFOUND

    @abi_case("nd_query_aggregates/sum_min_max_average")
    def test_aggregates(self, tmp_path: Any) -> None:
        """Aggregations run on numeric attributes of any project."""

        app: Nadata = numeric_project(tmp_path)
        boxes: Any = app.query("Box")

        assert boxes.sum_of("weight") == pytest.approx(7.5)
        assert boxes.min_of("weight") == pytest.approx(1.5)
        assert boxes.max_of("weight") == pytest.approx(3.5)
        assert boxes.average_of("weight") == pytest.approx(2.5)

        assert boxes.sum_of("count") == 10
        assert boxes.min_of("count") == 2
        assert boxes.max_of("count") == 5
        assert boxes.average_of("count") == pytest.approx(10.0 / 3.0)

    @abi_case("nd_query_aggregates/non_numeric_attribute")
    def test_aggregates_non_numeric(self) -> None:
        """A non-numeric attribute -> ND_ERR_TYPE."""

        app: Nadata = loaded_ex0()

        with pytest.raises(TypeError) as info:
            app.query("Human").sum_of("name")

        assert status_for(info.value) is NdStatus.ND_ERR_TYPE

    @abi_case("nd_query_to_list_ids/ordered_and_stable")
    def test_to_list_stable_ids(self) -> None:
        """to_list yields the same instances in the same order."""

        app: Nadata = loaded_ex0()

        first: list[Any] = app.query("Human").to_list()
        second: list[Any] = app.query("Human").to_list()

        assert first == second


# ------------------------------------------------------------------ #
# Persistence                                                         #
# ------------------------------------------------------------------ #


class TestPersistence:
    """nd_dump / nd_save_changes / nd_compact / nd_set_loop_limit."""

    @abi_case("nd_dump/file_must_not_exist")
    def test_dump_existing_file_is_value_error(self, tmp_path: Any) -> None:
        """dump() refuses to overwrite (snapshot semantics)."""

        storage_path: str = dumped_ex0(tmp_path)

        with pytest.raises(ValueError) as info:
            loaded_ex0().dump(storage_path=storage_path)

        assert status_for(info.value) is NdStatus.ND_ERR_VALUE

    @abi_case("nd_save_changes/delta_appended")
    def test_save_changes_delta(self, tmp_path: Any) -> None:
        """Incremental saves append only the delta; reopen sees it."""

        storage_path: str = dumped_ex0(tmp_path)
        app: Nadata = Nadata.open(storage_path=storage_path)

        app.create_instance("Human", {"name": "Zoe"}, symbol="zoe")
        appended: int = app.save_changes(storage_path=storage_path)

        assert appended >= 1

        reopened: Nadata = Nadata.open(storage_path=storage_path)

        assert reopened.count("Human") == EX0_HUMANS + 1
        assert reopened.get_attribute(symbol="zoe", name="name") == "Zoe"

    @abi_case("nd_save_changes/missing_file")
    def test_save_changes_missing_file(self, tmp_path: Any) -> None:
        """save_changes on a missing file -> ND_ERR_VALUE (dump first)."""

        app: Nadata = loaded_ex0()

        with pytest.raises(ValueError) as info:
            app.save_changes(storage_path=str(tmp_path / "never.ndb"))

        assert status_for(info.value) is NdStatus.ND_ERR_VALUE

    @abi_case("nd_compact/in_place_returns_counts")
    def test_compact_in_place(self, tmp_path: Any) -> None:
        """Compaction returns (before, after) and reopens cleanly."""

        storage_path: str = dumped_ex0(tmp_path)
        app: Nadata = Nadata.open(storage_path=storage_path)
        app.create_instance("Human", {"name": "Zoe"})
        app.save_changes(storage_path=storage_path)

        before: int
        after: int
        before, after = app.compact()

        assert after <= before

        reopened: Nadata = Nadata.open(storage_path=storage_path)

        assert reopened.count("Human") == EX0_HUMANS + 1

    @abi_case("nd_set_loop_limit/valid_value")
    def test_set_loop_limit_valid(self) -> None:
        """A higher fuel cap is accepted and persisted."""

        loaded_ex0().set_loop_limit(limit=100_000)

    @abi_case("nd_set_loop_limit/below_one")
    def test_set_loop_limit_invalid(self) -> None:
        """Sub-1 limits -> ND_ERR_VALUE (engine-validated)."""

        app: Nadata = loaded_ex0()

        for invalid in (0, -5):
            with pytest.raises(ValueError) as info:
                app.set_loop_limit(limit=invalid)

            assert status_for(info.value) is NdStatus.ND_ERR_VALUE


# ------------------------------------------------------------------ #
# Code execution                                                      #
# ------------------------------------------------------------------ #


class TestExecution:
    """nd_execute_schema / nd_execute_data / nd_execute / script files."""

    @abi_case("nd_execute_schema/live_merge")
    def test_execute_schema_live_merge(self) -> None:
        """New structs are immediately instantiable."""

        app: Nadata = loaded_ex0()

        app.execute_schema(
            'class Weapon { name: Str = "Sword"; }'
        )

        app.create_instance(
            struct_name="Weapon", attributes={"name": "Excalibur"},
            symbol="exc",
        )

        assert app.get_attribute(symbol="exc", name="name") == "Excalibur"

    @abi_case("nd_execute_data/instances_bound")
    def test_execute_data_binds_symbols(self) -> None:
        """Data source creates and binds instances at once."""

        app: Nadata = loaded_ex0()

        app.execute_schema('class Weapon { name: Str = "Sword"; }')
        app.execute_data('knife2 = Weapon(name = "Knife 2")')

        assert app.get_attribute(symbol="knife2", name="name") == "Knife 2"

    @abi_case("nd_execute/auto_detection")
    def test_execute_auto_detects(self) -> None:
        """execute() routes schema-looking text to the schema path."""

        app: Nadata = loaded_ex0()

        app.execute('class Shield { defense: Int32 = 5; }')

        assert "Shield" in app.struct_names

    @abi_case("nd_execute_script_file/params")
    def test_execute_script_file(self, tmp_path: Any) -> None:
        """#manual scripts take parameters and persist their effects."""

        app: Nadata
        folder: Any
        app, folder = session_project(tmp_path)

        app.execute_script_file(
            file_path=str(folder / "capture.nadata"),
            params={"v": 42, "name": "Zoe"},
        )

        assert app.get_attribute(symbol="captured", name="age") == 42
        assert app.get_attribute(symbol="captured", name="name") == "Zoe"


# ------------------------------------------------------------------ #
# Suite integrity                                                     #
# ------------------------------------------------------------------ #


class TestSuiteIntegrity:
    """Guards over the contract itself."""

    def test_case_ids_unique(self) -> None:
        """Every abi_case id in this module is unique (fail at import)."""

        ids: set[str] = registered_case_ids()

        assert len(ids) > 0

    def test_home_types_documented(self) -> None:
        """The unrepresentable home types are declared for bindings."""

        assert "DateTime" in HOME_TYPES_NOT_REPRESENTABLE
