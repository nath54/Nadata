"""
Tests for the resolution phase: tmp IR to registered database objects.
"""

# Import Modules
from __future__ import annotations
from typing import Any

import pytest

from src.core.types.types import TypeArray, TypeDict, UFloat32, UInt8
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime
from src.core.values.values import ValueDateTime, ValueStr
from src.core.types.types import TypeStr
from src.parser.nadata_loader.loader import NaDataLoader
from src.parser.nastruct_parser.nastruct_parser import NaStructParser
from src.parser.nastruct_parser.resolver import build_type_globals


class TestResolution:
    """
    Tests for resolving .nastruct sources into database tables.
    """

    def test_parse_folder_populates_database(self) -> None:
        """
        Tests that parse_folder registers all structs and functions.
        """

        parser: NaStructParser = NaStructParser()
        database = parser.parse_folder(folder_path="./examples/ex0_simple")

        assert set(database.tables_by_struct_name) == {
            "Entity", "Human", "Animal", "Cat", "ObjectType", "ObjectInstance"
        }
        assert set(database.functions) == {"calculate_age_difference", "is_adult"}

        # Every table has a type id and is indexed by type id too
        entity_table = database.tables_by_struct_name["Entity"]

        assert database.tables_by_type_id[entity_table.type_id] is entity_table

    def test_inheritance_and_attribute_types(self) -> None:
        """
        Tests resolved parents, attribute types, and inherited attributes.
        """

        parser: NaStructParser = NaStructParser()
        database = parser.parse_folder(folder_path="./examples/ex0_simple")

        entity = database.tables_by_struct_name["Entity"].struct
        human = database.tables_by_struct_name["Human"].struct

        # Inheritance chain
        assert human.parent is entity

        # Attribute types
        assert entity.attributes["name"].attr_type is not None
        assert "size" in human.get_all_attributes()

        # Struct typed arrays resolve to the actual struct object
        parents_attr = entity.attributes["parents"]

        assert isinstance(parents_attr.attr_type, TypeArray)
        assert parents_attr.attr_type.value_type is entity

        # Inherited attributes are visible on children
        assert "name" in human.get_all_attributes()
        assert "pets" in human.get_all_attributes()

    def test_default_values_evaluated(self) -> None:
        """
        Tests that default expressions are evaluated into raw values.
        """

        parser: NaStructParser = NaStructParser()
        database = parser.parse_folder(folder_path="./examples/ex0_simple")

        entity = database.tables_by_struct_name["Entity"].struct
        cat = database.tables_by_struct_name["Cat"].struct

        assert entity.attributes["gender"].default_value == "?"
        assert entity.attributes["birthplace"].default_value == ""
        assert entity.attributes["parents"].default_value == []

        # Color(...) call defaults evaluate to a Color instance
        assert cat.attributes["color"].default_value == Color(255, 255, 255)

        # DateTime() defaults evaluate to a DateTime instance
        birthday_default = entity.attributes["birthday"].default_value

        assert hasattr(birthday_default, "year")

    def test_methods_resolved(self) -> None:
        """
        Tests that methods carry resolved parameter and return types.
        """

        parser: NaStructParser = NaStructParser()
        database = parser.parse_folder(folder_path="./examples/ex0_simple")

        entity = database.tables_by_struct_name["Entity"].struct
        age = entity.methods["age"]

        assert isinstance(age.params["now"], type(UInt8)) or age.params["now"] is not None
        assert age.return_type is not None
        assert age.body_block is not None
        assert len(age.body_block.statements) == 1

        # Default parameter expressions stay lazy
        assert "now" in age.default_param_exprs

    def test_forward_references(self) -> None:
        """
        Tests that types used before their declaration resolve correctly.
        """

        parser: NaStructParser = NaStructParser()
        database = parser.parse_folder(folder_path="./examples/ex2_inheritance/structs")

        server = database.tables_by_struct_name["Server"].struct

        # connected_peers: Array[NetworkNode] declared before resolution order
        peers_attr = server.get_all_attributes()["connected_peers"]

        assert isinstance(peers_attr.attr_type, TypeArray)

        # WirelessRouter inherits from Router (empty struct body)
        wireless = database.tables_by_struct_name["WirelessRouter"].struct
        router = database.tables_by_struct_name["Router"].struct

        assert wireless.parent is router

    def test_self_referencing_type(self) -> None:
        """
        Tests that a struct referencing itself resolves without a cycle error.
        """

        parser: NaStructParser = NaStructParser()
        database = parser.parse_folder(folder_path="./examples/ex0_simple")

        entity = database.tables_by_struct_name["Entity"].struct

        assert isinstance(entity.attributes["parents"].attr_type, TypeArray)

    def test_duplicate_struct_rejected(self) -> None:
        """
        Tests that a duplicated struct name raises a SyntaxError at scan time.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="already defined"):
            parser.parse_file(content=(
                "class A {}\n"
                "class A {}\n"
            ))

    def test_unknown_attribute_type_rejected(self) -> None:
        """
        Tests that an unknown type name raises a TypeError at resolution.
        """

        parser: NaStructParser = NaStructParser()
        parser.parse_file(content=(
            "class A {\n"
            "    x: NotAType;\n"
            "}\n"
        ))

        with pytest.raises(TypeError, match="Unknown struct or type"):
            parser.resolve()

    def test_generic_arity_validated(self) -> None:
        """
        Tests that Array / Dict generics validate their argument count.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(TypeError, match="exactly two"):
            parser.parse_file(content=(
                "class A {\n"
                "    d: Dict[Str];\n"
                "}\n"
            ))
            parser.resolve()

    def test_resolved_dict_type(self) -> None:
        """
        Tests that Dict[K, V] resolves with key and value types.
        """

        parser: NaStructParser = NaStructParser()
        parser.parse_file(content=(
            "class A {\n"
            "    d: Dict[Str, Int32] = {};\n"
            "}\n"
        ))
        database = parser.resolve()

        attr = database.tables_by_struct_name["A"].struct.attributes["d"]

        assert isinstance(attr.attr_type, TypeDict)
        assert attr.attr_type.key_type is not None
        assert attr.default_value == {}


class TestRuntime:
    """
    End to end tests: resolved schemas driving real instance execution.
    """

    def setup_method(self) -> None:
        """
        Parses ex0 and creates two Human instances for the runtime tests.
        """

        parser: NaStructParser = NaStructParser()
        self.database = parser.parse_folder(folder_path="./examples/ex0_simple")
        self.globals_dict = build_type_globals()

        human_struct = self.database.tables_by_struct_name["Human"].struct

        self.tom = self.database.create_instance(
            struct=human_struct,
            symbol="tom",
            attributes={
                "name": ValueStr(value_type=TypeStr(), value="Tom"),
                "birthday": ValueDateTime(value=DateTime(d=12, m=4, y=2003)),
            }
        )
        self.jean = self.database.create_instance(
            struct=human_struct,
            symbol="jean",
            attributes={
                "name": ValueStr(value_type=TypeStr(), value="Jean"),
                "birthday": ValueDateTime(value=DateTime(d=12, m=4, y=1972)),
            }
        )

    def test_inherited_defaults_reachable(self) -> None:
        """
        Tests that inherited attribute defaults resolve at runtime.
        """

        assert self.tom.get_attribute("gender").get() == "?"
        assert self.tom.get_attribute("name").get() == "Tom"

    def test_method_execution_with_lazy_default_param(self) -> None:
        """
        Tests method execution with a default parameter (DateTime.now()).
        """

        age_value = self.tom.call_method(
            name="age",
            args=[],
            globals_dict=self.globals_dict
        )

        assert hasattr(age_value.get(), "year")

    def test_standalone_functions(self) -> None:
        """
        Tests standalone function execution through the database.
        """

        is_adult = self.database.functions["is_adult"]

        assert is_adult.execute(
            self_instance=None,
            args=[self.tom],
            globals_dict=self.globals_dict
        ).get() is True

    def test_inherited_method_with_if_else(self) -> None:
        """
        Tests the inherited older() method executing if / else branches.
        """

        entity_struct = self.database.tables_by_struct_name["Entity"].struct
        older = entity_struct.methods["older"]

        # Jean (1972) is older than Tom (2003)
        assert older.execute(
            self_instance=self.jean,
            args=[self.tom],
            globals_dict=self.globals_dict
        ).get() is True

    def test_type_checking_rejects_wrong_struct(self) -> None:
        """
        Tests that passing a Cat to a Human parameter raises TypeError.
        """

        cat_struct = self.database.tables_by_struct_name["Cat"].struct
        cat = self.database.create_instance(struct=cat_struct, symbol="felix")

        is_adult = self.database.functions["is_adult"]

        with pytest.raises(TypeError, match="Expected instance of Struct 'Human'"):
            is_adult.execute(
                self_instance=None,
                args=[cat],
                globals_dict=self.globals_dict
            )


class TestExplicitSelf:
    """
    Tests the explicit self rule: attributes via 'self.', no shadowing.
    """

    def test_explicit_self_resolves_attribute(self) -> None:
        """
        Tests that 'self.attribute' resolves during method execution.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class Counter {\n"
            "    count: Int32 = 0;\n"
            "\n"
            "    @doubled() -> Int32 {\n"
            "        return self.count * 2;\n"
            "    }\n"
            "}\n"
        ))

        database = parser.resolve()

        loader: NaDataLoader = NaDataLoader(database=database)
        loader.load_file(content='c = Counter(count = 21)')

        counter: Any = database.get_by_symbol(name="c")

        assert counter.call_method("doubled").get() == 42

    def test_bare_attribute_name_is_undefined(self) -> None:
        """
        Tests that bare attribute names no longer resolve in method bodies.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class Counter {\n"
            "    count: Int32 = 0;\n"
            "\n"
            "    @doubled() -> Int32 {\n"
            "        return count * 2;\n"
            "    }\n"
            "}\n"
        ))

        database = parser.resolve()

        loader: NaDataLoader = NaDataLoader(database=database)
        loader.load_file(content='c = Counter()')

        counter: Any = database.get_by_symbol(name="c")

        with pytest.raises(NameError):
            counter.call_method("doubled")

    def test_parameter_shadowing_attribute_rejected(self) -> None:
        """
        Tests that a parameter named like an attribute is a resolution error.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="shadows an attribute"):
            parser.parse_folder = None  # type: ignore[assignment]

            parser.parse_file(content=(
                "class A {\n"
                "    name: Str;\n"
                "\n"
                "    @m(name: Int32) -> Int32 {\n"
                "        return name;\n"
                "    }\n"
                "}\n"
            ))

            parser.resolve()

    def test_local_shadowing_attribute_rejected(self) -> None:
        """
        Tests that a local variable named like an attribute is rejected.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="shadows an attribute"):
            parser.parse_file(content=(
                "class A {\n"
                "    name: Str;\n"
                "\n"
                "    @m() -> Int32 {\n"
                "        Int32 name = 5;\n"
                "        return name;\n"
                "    }\n"
                "}\n"
            ))

            parser.resolve()

    def test_inherited_attribute_shadowing_rejected(self) -> None:
        """
        Tests that shadowing a parent attribute is rejected too.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="shadows an attribute"):
            parser.parse_file(content=(
                "class A {\n"
                "    name: Str;\n"
                "}\n"
                "\n"
                "class B(A) {\n"
                "    @m(name: Int32) -> Int32 {\n"
                "        return name;\n"
                "    }\n"
                "}\n"
            ))

            parser.resolve()

    def test_disjoint_names_are_accepted(self) -> None:
        """
        Tests that parameters reusing nothing attribute-like still resolve.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content=(
            "class A {\n"
            "    name: Str;\n"
            "\n"
            "    @m(times: Int32) -> Int32 {\n"
            "        Int32 doubled = times * 2;\n"
            "        return doubled;\n"
            "    }\n"
            "}\n"
        ))

        database = parser.resolve()

        assert "m" in database.tables_by_struct_name["A"].struct.methods

    def test_self_undefined_in_scripts(self) -> None:
        """
        Tests that 'self' does not resolve outside method bodies.
        """

        parser: NaStructParser = NaStructParser()

        parser.parse_file(content="class A { name: Str; }\n")

        database = parser.resolve()

        loader: NaDataLoader = NaDataLoader(database=database)

        with pytest.raises(NameError):
            loader.load_file(content='x = self')

