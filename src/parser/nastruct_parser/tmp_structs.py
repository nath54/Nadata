"""
TODO
"""

# Import Modules
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


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
        attribute_default_value (Optional[Any]): Raw default value expression
            string, or None if the attribute has no default.
    """

    attribute_name: str
    attribute_type: str
    attribute_default_value: Optional[str] = None

    dependencies: set[str] = field(default_factory=set)


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
