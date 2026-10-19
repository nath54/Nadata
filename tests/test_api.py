"""
Tests for the Nadata high level API facade.

Covers project loading, instance access with plain Python values, function
and method calls, polymorphic queries, instance creation, and persistence.
"""

# Import Modules
import os
from typing import Any

import pytest

from src.api import Nadata


class TestNadataLoading:
    """
    Tests facade construction from projects, schemas, and database files.
    """

    def test_load_project(self) -> None:
        """
        Tests loading a full example project folder.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")

        assert "Human" in app.struct_names
        assert "tom" in app.symbol_names
        assert "is_adult" in app.function_names

    def test_load_single_schema_file(self) -> None:
        """
        Tests loading a single .nastruct schema file (no instances).
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple/structs/structs.nastruct")

        assert "Human" in app.struct_names
        assert app.symbol_names == []

    def test_load_data_file_alone_raises(self) -> None:
        """
        Tests that loading a .nadata file without its schema raises.
        """

        with pytest.raises(ValueError, match="alone"):
            Nadata.load(path="./examples/ex0_simple/data/data1.nadata")

    def test_load_unsupported_extension_raises(self) -> None:
        """
        Tests that unsupported path extensions raise ValueError.
        """

        with pytest.raises(ValueError, match="Unsupported"):
            Nadata.load(path="./examples/does_not_exist.txt")


class TestNadataRuntime:
    """
    Tests runtime operations through plain Python values.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example for the runtime tests.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_get_and_get_attribute(self) -> None:
        """
        Tests symbol lookup and raw attribute access.
        """

        assert self.app.get_attribute(symbol="tom", name="name") == "Tom"
        assert self.app.get_attribute(symbol="white_cat", name="name") == "White cat"

    def test_get_unknown_symbol_raises(self) -> None:
        """
        Tests that unknown symbols raise KeyError.
        """

        with pytest.raises(KeyError, match="Unknown symbol"):
            self.app.get(symbol="nobody")

    def test_call_function(self) -> None:
        """
        Tests standalone function calls with plain Python arguments.
        """

        tom = self.app.get(symbol="tom")

        assert self.app.call_function("is_adult", tom) is True

    def test_call_function_by_kwargs(self) -> None:
        """
        Tests standalone function calls with keyword arguments.
        """

        tom = self.app.get(symbol="tom")

        result: Any = self.app.call_function(
            "calculate_age_difference",
            h1=tom,
            h2=tom,
        )

        assert hasattr(result, "year")

    def test_call_function_unknown_raises(self) -> None:
        """
        Tests that unknown function names raise KeyError.
        """

        with pytest.raises(KeyError, match="Unknown function"):
            self.app.call_function("no_such_function")

    def test_call_method(self) -> None:
        """
        Tests method calls on symbol bound instances.
        """

        result: Any = self.app.call_method("tom", "age")

        assert hasattr(result, "year")

    def test_call_method_with_argument(self) -> None:
        """
        Tests method calls with a plain float argument (ex4 pipeline).
        """

        app: Nadata = Nadata.load(path="./examples/ex4_expressions")

        assert app.call_method("pipe_standard", "apply_transform", 10.0) == 34.0

    def test_call_method_unknown_raises(self) -> None:
        """
        Tests that unknown method names raise KeyError.
        """

        with pytest.raises(KeyError, match="no method"):
            self.app.call_method("tom", "no_such_method")

    def test_query_and_count_polymorphic(self) -> None:
        """
        Tests polymorphic queries counting subclass instances.
        """

        # Entity + Human + Animal + Cat tables = 3 humans + 1 cat
        assert self.app.count("Entity") == 4
        assert self.app.count("Human") == 3

    def test_query_with_where(self) -> None:
        """
        Tests chained where() filtering with a plain callable predicate.
        """

        parisians = self.app.query("Human").where(
            lambda inst: inst.get_attribute("birthplace").get() == "Paris"
        )

        assert len(parisians.to_list()) == 3

    def test_set_attribute(self) -> None:
        """
        Tests setting attributes with plain Python values.
        """

        self.app.set_attribute(symbol="tom", name="birthplace", value="Lyon")

        assert self.app.get_attribute(symbol="tom", name="birthplace") == "Lyon"

    def test_set_attribute_type_validated(self) -> None:
        """
        Tests that set_attribute validates against the schema type.
        """

        with pytest.raises(TypeError):
            self.app.set_attribute(symbol="tom", name="birthday", value="not a date")

    def test_create_instance(self) -> None:
        """
        Tests instance creation with plain Python attribute values.
        """

        instance = self.app.create_instance(
            struct_name="Human",
            attributes={"name": "New Guy"},
            symbol="new_guy",
        )

        assert instance.get_attribute("name").get() == "New Guy"
        assert "new_guy" in self.app.symbol_names
        assert self.app.count("Human") == 4

    def test_create_instance_unknown_struct_raises(self) -> None:
        """
        Tests that creating an instance of an unknown struct raises.
        """

        with pytest.raises(KeyError, match="Unknown struct"):
            self.app.create_instance(struct_name="Nope")

    def test_summary_and_counts(self) -> None:
        """
        Tests the summary rendering and instance counting.
        """

        summary: str = self.app.summary()

        assert "Structs (6)" in summary
        assert "Human, extends Entity" in summary
        assert self.app.total_instance_count() == 6


class TestNadataPersistence:
    """
    Tests dump / open round trips through the facade.
    """

    def test_dump_and_open(self, tmp_path: Any) -> None:
        """
        Tests that a dumped database reopens with all data intact.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        storage_path: str = str(tmp_path / "api_test.ndb")

        app.dump(storage_path=storage_path)
        reopened: Nadata = Nadata.open(storage_path=storage_path)

        assert reopened.get_attribute(symbol="tom", name="name") == "Tom"
        assert reopened.count("Human") == 3
        assert sorted(reopened.symbol_names) == sorted(app.symbol_names)

        # Created before dump: visible after reopen
        app.create_instance(
            struct_name="Human",
            attributes={"name": "Late Guy"},
            symbol="late_guy",
        )
        app.dump(storage_path=str(tmp_path / "second.ndb"))
        second: Nadata = Nadata.open(storage_path=str(tmp_path / "second.ndb"))

        assert second.get_attribute(symbol="late_guy", name="name") == "Late Guy"

    def test_load_delegates_ndb_to_open(self, tmp_path: Any) -> None:
        """
        Tests that load() on a .ndb file opens the persisted database.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        storage_path: str = str(tmp_path / "delegated.ndb")

        app.dump(storage_path=storage_path)
        reopened: Nadata = Nadata.load(path=storage_path)

        assert reopened.get_attribute(symbol="tom", name="name") == "Tom"

    def test_dump_to_existing_path_raises(self, tmp_path: Any) -> None:
        """
        Tests that dumping over an existing file raises ValueError.
        """

        app: Nadata = Nadata.load(path="./examples/ex0_simple")
        storage_path: str = str(tmp_path / "exists.ndb")

        storage_file = open(storage_path, "w")
        storage_file.close()

        with pytest.raises(ValueError, match="already exists"):
            app.dump(storage_path=storage_path)

        os.remove(storage_path)


class TestNadataCodeExecution:
    """
    Tests executing raw .nastruct and .nadata source strings.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example for the execution tests.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_execute_schema_adds_struct(self) -> None:
        """
        Tests that schema source registers a new struct with methods.
        """

        self.app.execute_schema(
            schema_text=(
                'class Weapon {\n'
                '    name: Str = "Sword";\n'
                '    damage: Int32 = 10;\n'
                '\n'
                '    @describe() -> Str {\n'
                '        return self.name;\n'
                '    }\n'
                '}'
            )
        )

        assert "Weapon" in self.app.struct_names

        excalibur = self.app.create_instance(
            struct_name="Weapon",
            attributes={"name": "Excalibur", "damage": 99},
            symbol="excalibur",
        )

        assert excalibur.get_attribute("damage").get() == 99
        assert self.app.call_method("excalibur", "describe") == "Excalibur"

    def test_execute_data_creates_instances(self) -> None:
        """
        Tests that data source creates and binds instances.
        """

        self.app.execute_data(data_text='wanderer = Human(name = "Wanderer")')

        assert self.app.get_attribute(symbol="wanderer", name="name") == "Wanderer"
        assert self.app.count("Human") == 4

    def test_execute_data_runs_script_statements(self, capsys: Any) -> None:
        """
        Tests that script statements (print) execute during execute_data.
        """

        self.app.execute_data(
            data_text='print("Hello from", tom.name);'
        )

        assert "Hello from Tom" in capsys.readouterr().out

    def test_execute_auto_detects_schema(self) -> None:
        """
        Tests that execute() routes class declarations to the schema parser.
        """

        self.app.execute(text="class Shield { defense: Int32 = 5; }")

        assert "Shield" in self.app.struct_names

    def test_execute_auto_detects_data(self) -> None:
        """
        Tests that execute() routes plain statements to the data loader.
        """

        self.app.execute(text='drifter = Human(name = "Drifter")')

        assert self.app.get_attribute(symbol="drifter", name="name") == "Drifter"

    def test_execute_data_unknown_struct_raises(self) -> None:
        """
        Tests that data referencing an unknown struct raises NameError.
        """

        with pytest.raises(NameError):
            self.app.execute_data(data_text="x = Missing(name = \"x\")")

    def test_executed_code_is_queryable_and_persistable(self, tmp_path: Any) -> None:
        """
        Tests that string executed content survives a dump / open cycle.
        """

        self.app.execute(text='class Gem { carat: Int32 = 1; }')
        self.app.execute(text='big_gem = Gem(carat = 5)')

        storage_path: str = str(tmp_path / "exec.ndb")

        self.app.dump(storage_path=storage_path)
        reopened: Nadata = Nadata.open(storage_path=storage_path)

        assert "Gem" in reopened.struct_names
        assert reopened.get_attribute(symbol="big_gem", name="carat") == 5
