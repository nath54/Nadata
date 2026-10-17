"""
Unit tests for Nadata core types, values, and struct schema models.
"""

# Import Modules
from __future__ import annotations
from typing import Any

import pytest

from src.core.struct.struct import Struct
from src.core.struct.struct_method import StructMethod
from src.core.struct.struct_attribute import StructAttribute
from src.core.types.types import (
    Int4,
    Int8,
    TypeStr,
    TypeInt,
    TypeBool,
    TypeDict,
    TypeArray,
    TypeFloat,
)
from src.core.values.values import (
    ValueStr,
    ValueInt,
    ValueBool,
    ValueDict,
    ValueArray,
    ValueFloat,
)


def test_type_int_bounds() -> None:
    """
    Tests that TypeInt correctly validates bounded values and rejects invalid values.
    """

    t_int: TypeInt = Int4

    # Valid values for 4-bit signed [-8, 7]
    t_int.check_value(value=7)
    t_int.check_value(value=-8)

    # Out of bounds
    with pytest.raises(ValueError):
        t_int.check_value(value=8)

    with pytest.raises(ValueError):
        t_int.check_value(value=-9)

    # Invalid type
    with pytest.raises(TypeError):
        t_int.check_value(value="not_int")


def test_value_int_serialization() -> None:
    """
    Tests ValueInt creation and lazy bit serialization.
    """

    val: ValueInt = ValueInt(value_type=Int4, value=5)

    assert val.get() == 5
    assert val.to_bits().to_bin_str() == "0101"

    # Test negative two's complement in Int4 (-3 -> 1101)
    val_neg: ValueInt = ValueInt(value_type=Int4, value=-3)

    assert val_neg.get() == -3
    assert val_neg.to_bits().to_bin_str() == "1101"


def test_type_float_and_value() -> None:
    """
    Tests TypeFloat and ValueFloat validation and bit conversion.
    """

    t_f32: TypeFloat = TypeFloat(num_bits=32, num_bits_exponent=8, signed=True)
    val_f: ValueFloat = ValueFloat(value_type=t_f32, value=1.5)

    assert val_f.get() == 1.5
    assert len(val_f.to_bits()) == 32


def test_containers_types_and_values() -> None:
    """
    Tests TypeArray, TypeDict, ValueArray, and ValueDict.
    """

    t_arr: TypeArray = TypeArray(value_type=TypeStr())
    val_arr: ValueArray = ValueArray(value_type=t_arr, elements=["a", "b", "c"])

    assert len(val_arr) == 3
    assert val_arr.get() == ["a", "b", "c"]

    t_dict: TypeDict = TypeDict(value_type=Int8)
    val_dict: ValueDict = ValueDict(
        value_type=t_dict,
        entries={"x": 10, "y": -20}
    )

    assert len(val_dict) == 2
    assert val_dict.get()["x"] == 10

    # Test ValueStr and ValueBool
    val_str: ValueStr = ValueStr(value_type=TypeStr(), value="hello")
    val_bool: ValueBool = ValueBool(value_type=TypeBool(), value=True)

    assert val_str.get() == "hello"
    assert val_bool.get() is True


def test_struct_inheritance_and_attributes() -> None:
    """
    Tests Struct hierarchy, inheritance, and attribute resolution.
    """

    # Base entity struct
    entity: Struct = Struct(name="Entity")
    attr_name: StructAttribute = StructAttribute(name="name", attr_type=TypeStr())
    attr_gender: StructAttribute = StructAttribute(
        name="gender",
        attr_type=TypeStr(),
        default_value="?",
        has_default=True
    )
    entity.add_attribute(attribute=attr_name)
    entity.add_attribute(attribute=attr_gender)

    # Derived human struct
    human: Struct = Struct(name="Human", parent=entity)
    attr_weight: StructAttribute = StructAttribute(
        name="weight",
        attr_type=TypeFloat(num_bits=32)
    )
    human.add_attribute(attribute=attr_weight)

    # Check inheritance
    assert human.is_subclass_of(other=entity)
    assert not entity.is_subclass_of(other=human)

    # Check attribute resolution
    assert human.get_attribute(name="weight") is attr_weight
    assert human.get_attribute(name="name") is attr_name

    # Check all attributes combined
    all_attrs: dict[str, Any] = human.get_all_attributes()

    assert "name" in all_attrs
    assert "gender" in all_attrs
    assert "weight" in all_attrs


def test_struct_method_registration() -> None:
    """
    Tests StructMethod definition and registration on a struct.
    """

    entity: Struct = Struct(name="Entity")
    method: StructMethod = StructMethod(
        name="is_alive",
        params={},
        return_type=TypeBool(),
        body_expression="state > 0;"
    )
    entity.add_method(method=method)

    resolved: StructMethod | None = entity.get_method(name="is_alive")

    assert resolved is not None
    assert resolved.name == "is_alive"
    assert repr(resolved) == "@is_alive() -> Bool"
