"""
Intermediate representation dataclasses for .nastruct parsing.

These structures hold raw parsed data between the scanning phase and the
type resolution phase.
"""

# Import Modules
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TmpAttribute:
    """
    Intermediate representation of an attribute or parameter declaration.

    Holds raw parsed data before type resolution; the type is kept as its
    source string and the default value as its raw expression string, until
    the resolution phase converts them into runtime Type and Value objects.

    Also reused to represent method/function parameters.

    Attributes:
        attribute_name (str): Name of the attribute or parameter.
        attribute_type (str): Raw type annotation string (e.g. 'Array[Entity]').
        attribute_default_value (Optional[str]): Raw default value expression
            string, or None if the attribute has no default.
    """

    attribute_name: str
    attribute_type: str
    attribute_default_value: Optional[str] = None

    dependencies: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the attribute to a plain JSON serializable dict.

        Returns:
            dict[str, Any]: Plain dict representation with dependencies as a
                sorted list.
        """

        return {
            "attribute_name": self.attribute_name,
            "attribute_type": self.attribute_type,
            "attribute_default_value": self.attribute_default_value,
            "dependencies": sorted(self.dependencies),
        }


@dataclass
class TmpFunction:
    """
    Intermediate representation of a method or standalone function.

    Holds raw parsed data before resolution: parameter declarations, the raw
    return type string, and the raw body text span. Expressions are NOT parsed
    at this stage (see the resolution phase).

    Attributes:
        fn_name (str): Name of the function or method.
        fn_output_type (str): Raw return type annotation string.
        fn_argument (dict[str, TmpAttribute]): Parameters by name.
        fn_body (str): Raw body text between the braces.
    """

    fn_name: str
    fn_output_type: str
    fn_argument: dict[str, TmpAttribute] = field(default_factory=dict)
    fn_body: str = ""

    dependencies: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the function to a plain JSON serializable dict.

        Returns:
            dict[str, Any]: Plain dict representation with parameters and
                dependencies serialized recursively.
        """

        return {
            "fn_name": self.fn_name,
            "fn_output_type": self.fn_output_type,
            "fn_argument": {
                name: argument.to_dict()
                for name, argument in self.fn_argument.items()
            },
            "fn_body": self.fn_body,
            "dependencies": sorted(self.dependencies),
        }


@dataclass
class TmpStruct:
    """
    Intermediate representation of a struct schema.

    Holds raw parsed data before resolution: the struct name, its optional
    parent name, its attribute declarations, and its method declarations.

    Attributes:
        struct_name (str): Name of the struct.
        struct_parent (Optional[str]): Name of the parent struct, if any.
        attributes (dict[str, TmpAttribute]): Attribute declarations by name.
        methods (dict[str, TmpFunction]): Method declarations by name.
    """

    struct_name: str
    struct_parent: Optional[str] = None

    attributes: dict[str, TmpAttribute] = field(default_factory=dict)
    methods: dict[str, TmpFunction] = field(default_factory=dict)

    dependencies: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the struct to a plain JSON serializable dict.

        Returns:
            dict[str, Any]: Plain dict representation with attributes,
                methods, and dependencies serialized recursively.
        """

        return {
            "struct_name": self.struct_name,
            "struct_parent": self.struct_parent,
            "attributes": {
                name: attribute.to_dict()
                for name, attribute in self.attributes.items()
            },
            "methods": {
                name: method.to_dict()
                for name, method in self.methods.items()
            },
            "dependencies": sorted(self.dependencies),
        }
