"""
Tests for the language built-ins: entity ids, type(), virtual members, and
schema introspection.
"""

# Import Modules
from __future__ import annotations
from typing import Any

import pytest

from src.api import Nadata


class TestEntityIdMember:
    """
    Tests the .id virtual member on struct instances.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example for the built-in tests.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_id_returns_instance_index(self, capsys: Any) -> None:
        """
        Tests that .id resolves to the per-struct instance index.
        """

        self.app.execute_data(data_text="print(tom.id);")

        assert "0" in capsys.readouterr().out

    def test_id_is_usable_in_expressions(self, capsys: Any) -> None:
        """
        Tests that .id participates in comparisons and arithmetic.
        """

        self.app.execute_data(data_text="print(tom.id == 0);")

        assert "True" in capsys.readouterr().out


class TestTypeBuiltin:
    """
    Tests the type() builtin and struct attribute type access.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example for the built-in tests.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_type_of_instance(self, capsys: Any) -> None:
        """
        Tests that type() on an instance returns its struct type.
        """

        self.app.execute_data(data_text="print(type(tom));")

        assert "<Type Human>" in capsys.readouterr().out

    def test_type_of_value(self, capsys: Any) -> None:
        """
        Tests that type() on a plain value returns its schema type.
        """

        self.app.execute_data(data_text="print(type(42));")

        assert "<Type Int64>" in capsys.readouterr().out

    def test_type_of_class_attribute(self, capsys: Any) -> None:
        """
        Tests that Struct.attribute yields the attribute's declared type.
        """

        self.app.execute_data(data_text="print(type(Human.birthday));")

        assert "<Type DateTime>" in capsys.readouterr().out

    def test_direct_class_attribute_access(self, capsys: Any) -> None:
        """
        Tests that Struct.attribute itself is the declared type.
        """

        self.app.execute_data(data_text="print(Human.birthday);")

        assert "<Type DateTime>" in capsys.readouterr().out

    def test_undeclared_class_attribute_raises(self) -> None:
        """
        Tests that accessing an undeclared attribute on a type raises.
        """

        with pytest.raises(AttributeError, match="not declared"):
            self.app.execute_data(data_text="print(Human.nope);")


class TestVirtualMembers:
    """
    Tests list_attr / list_methods / getattr / setattr on instances and types.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example for the built-in tests.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_list_attr_on_instance(self, capsys: Any) -> None:
        """
        Tests that instance list_attr includes inherited attributes.
        """

        self.app.execute_data(data_text="print(tom.list_attr());")

        out: str = capsys.readouterr().out

        assert "name" in out
        assert "size" in out

    def test_list_methods_on_type(self, capsys: Any) -> None:
        """
        Tests that type list_methods includes inherited methods.
        """

        self.app.execute_data(data_text="print(Human.list_methods());")

        out: str = capsys.readouterr().out

        assert "age" in out
        assert "older" in out

    def test_list_attr_on_type(self, capsys: Any) -> None:
        """
        Tests that list_attr works on struct types too.
        """

        self.app.execute_data(data_text="print(ObjectType.list_attr());")

        assert "description" in capsys.readouterr().out

    def test_getattr_reads_schema_attribute(self, capsys: Any) -> None:
        """
        Tests that getattr returns the current attribute value.
        """

        self.app.execute_data(data_text='print(tom.getattr("name"));')

        assert "Tom" in capsys.readouterr().out

    def test_getattr_unknown_attribute_raises(self) -> None:
        """
        Tests that getattr on a name outside the schema raises.
        """

        with pytest.raises(AttributeError, match="not declared"):
            self.app.execute_data(data_text='tom.getattr("nope");')

    def test_setattr_writes_schema_attribute(self) -> None:
        """
        Tests that setattr updates the instance and is visible to the API.
        """

        self.app.execute_data(data_text='tom.setattr("birthplace", "Lyon");')

        assert self.app.get_attribute(symbol="tom", name="birthplace") == "Lyon"

    def test_setattr_unknown_attribute_raises(self) -> None:
        """
        Tests that setattr on a name outside the schema raises.
        """

        with pytest.raises(AttributeError, match="not declared|Cannot set"):
            self.app.execute_data(data_text='tom.setattr("nope", 1);')

    def test_type_getattr_returns_attribute_type(self, capsys: Any) -> None:
        """
        Tests that getattr on a type returns the attribute's declared type.
        """

        self.app.execute_data(data_text='print(Human.getattr("size"));')

        assert "UFloat16" in capsys.readouterr().out


class TestSchemaIntrospection:
    """
    Tests the schema_classes() and schema_functions() built-ins.
    """

    def setup_method(self) -> None:
        """
        Loads the ex0 example for the built-in tests.
        """

        self.app: Nadata = Nadata.load(path="./examples/ex0_simple")

    def test_schema_classes_lists_all_structs(self, capsys: Any) -> None:
        """
        Tests that schema_classes lists every registered struct.
        """

        self.app.execute_data(data_text="print(schema_classes());")

        out: str = capsys.readouterr().out

        assert "<Type Entity>" in out
        assert "<Type Human>" in out
        assert "<Type Cat>" in out

    def test_schema_functions_lists_all_functions(self, capsys: Any) -> None:
        """
        Tests that schema_functions lists every registered function.
        """

        self.app.execute_data(data_text="print(schema_functions());")

        out: str = capsys.readouterr().out

        assert "calculate_age_difference" in out
        assert "is_adult" in out
