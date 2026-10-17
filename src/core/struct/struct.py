"""
Struct definition for Nadata schemas.

Represents an entity class schema, including its name, parent inheritance,
declared attributes, and declared methods.
"""

# Import Modules
from __future__ import annotations
from typing import Any

from src.core.struct.struct_method import StructMethod
from src.core.struct.struct_attribute import StructAttribute


class Struct:
    """
    Represents a struct schema definition (analogous to a class).

    Attributes:
        name (str): Identifier name of the struct (e.g. 'Human', 'Entity').
        parent (Struct | None): Optional base struct this struct inherits from.
        attributes (dict[str, StructAttribute]): Attributes defined in this struct.
        methods (dict[str, StructMethod]): Methods defined in this struct.
    """

    def __init__(
        self,
        name: str,
        parent: Struct | None = None
    ) -> None:
        """
        Initializes the Struct instance.

        Args:
            name (str): Name of the struct.
            parent (Struct | None, optional): Parent struct for inheritance. Defaults to None.
        """

        self.name: str = name
        self.parent: Struct | None = parent
        self.attributes: dict[str, StructAttribute] = {}
        self.methods: dict[str, StructMethod] = {}

    def add_attribute(self, attribute: StructAttribute) -> None:
        """
        Adds an attribute definition to this struct.

        Args:
            attribute (StructAttribute): Attribute to register.

        Raises:
            ValueError: If an attribute with the same name already exists in this struct.
        """

        # Ensure no attribute duplication within the same struct
        if attribute.name in self.attributes:
            raise ValueError(
                f"Attribute '{attribute.name}' is already defined on struct '{self.name}'"
            )

        self.attributes[attribute.name] = attribute

    def add_method(self, method: StructMethod) -> None:
        """
        Adds a method definition to this struct.

        Args:
            method (StructMethod): Method to register.

        Raises:
            ValueError: If a method with the same name already exists in this struct.
        """

        # Ensure no method duplication within the same struct
        if method.name in self.methods:
            raise ValueError(
                f"Method '@{method.name}' is already defined on struct '{self.name}'"
            )

        self.methods[method.name] = method

    def get_attribute(self, name: str) -> StructAttribute | None:
        """
        Resolves an attribute by name, looking in this struct and its parents.

        Args:
            name (str): Name of the attribute.

        Returns:
            StructAttribute | None: Attribute definition if found, None otherwise.
        """

        # Check local attributes first
        if name in self.attributes:
            return self.attributes[name]

        # Recursively search parent hierarchy
        if self.parent is not None:
            return self.parent.get_attribute(name=name)

        return None

    def get_all_attributes(self) -> dict[str, StructAttribute]:
        """
        Collects all inherited and locally defined attributes.

        Returns:
            dict[str, StructAttribute]: Dictionary of all combined attributes.
        """

        combined: dict[str, StructAttribute] = {}

        # Merge parent attributes first
        if self.parent is not None:
            combined.update(self.parent.get_all_attributes())

        # Local attributes take precedence or add to the collection
        combined.update(self.attributes)

        return combined

    def get_method(self, name: str) -> StructMethod | None:
        """
        Resolves a method by name, looking in this struct and its parents.

        Args:
            name (str): Name of the method.

        Returns:
            StructMethod | None: Method definition if found, None otherwise.
        """

        # Check local methods first
        if name in self.methods:
            return self.methods[name]

        # Recursively search parent hierarchy
        if self.parent is not None:
            return self.parent.get_method(name=name)

        return None

    def is_subclass_of(self, other: Struct) -> bool:
        """
        Checks if this struct inherits from another struct.

        Args:
            other (Struct): Candidate ancestor struct.

        Returns:
            bool: True if this struct is or inherits from other.
        """

        # Identity check
        if self == other:
            return True

        # Check parent link
        if self.parent is not None:
            return self.parent.is_subclass_of(other=other)

        return False

    def check_value(self, value: Any) -> None:
        """
        Validates that a given value is an instance conforming to this struct.

        Args:
            value (Any): Instance object to validate.

        Raises:
            TypeError: If value is not an instance conforming to this struct.
        """

        # Check struct attribute on candidate instance
        if hasattr(value, "struct"):
            candidate_struct: Any = getattr(value, "struct")

            if hasattr(candidate_struct, "is_subclass_of"):
                if candidate_struct.is_subclass_of(other=self):
                    return

        raise TypeError(
            f"Expected instance of Struct '{self.name}', got '{type(value).__name__}'"
        )

    def __repr__(self) -> str:
        """
        Returns string representation of the Struct.

        Returns:
            str: Struct signature representation.
        """

        if self.parent is not None:
            return f"Struct {self.name}({self.parent.name})"

        return f"Struct {self.name}"
