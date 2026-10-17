"""
Struct attribute definition for Nadata schemas.

Represents an individual field declared within a struct, including its name,
type specification, and optional default value.
"""

# Import Modules
from __future__ import annotations
from typing import Any

from src.core.types.types import Type


class StructAttribute:
    """
    Represents an attribute/field within a struct definition.

    Attributes:
        name (str): Identifier name of the attribute.
        attr_type (Type): Schema type associated with this attribute.
        default_value (Any): Default value if not specified during instantiation.
        has_default (bool): Whether a default value was explicitly provided.
    """

    def __init__(
        self,
        name: str,
        attr_type: Type,
        default_value: Any = None,
        has_default: bool = False
    ) -> None:
        """
        Initializes the StructAttribute instance.

        Args:
            name (str): Attribute name identifier.
            attr_type (Type): Attribute schema type.
            default_value (Any, optional): Optional default value. Defaults to None.
            has_default (bool, optional): Whether a default was provided. Defaults to False.
        """

        self.name: str = name
        self.attr_type: Type = attr_type
        self.default_value: Any = default_value
        self.has_default: bool = has_default

        # Validate default value against attribute type if present
        if has_default and default_value is not None:
            attr_type.check_value(value=default_value)

    def validate_value(self, value: Any) -> None:
        """
        Validates that a given value matches this attribute's type.

        Args:
            value (Any): Value to validate.

        Raises:
            TypeError: If the value type does not match.
            ValueError: If the value violates bounds or constraints.
        """

        # Delegate validation to the attribute type
        self.attr_type.check_value(value=value)

    def __repr__(self) -> str:
        """
        Returns string representation of the StructAttribute.

        Returns:
            str: Representation showing name, type, and default if present.
        """

        if self.has_default:
            return f"{self.name}: {repr(self.attr_type)} = {repr(self.default_value)}"

        return f"{self.name}: {repr(self.attr_type)}"
