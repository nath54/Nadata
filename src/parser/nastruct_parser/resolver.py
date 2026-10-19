"""
Resolution phase for .nastruct parsing.

Converts the intermediate representation (TmpStruct / TmpFunction) into core
Struct, StructAttribute, and StructMethod objects registered in the database:
TmpType trees are resolved into Type instances, default expressions are
evaluated into typed values, and bodies are attached as statement blocks.
"""

# Import Modules
from __future__ import annotations
from typing import Any, Callable, Optional, cast

from src.core.database.database import NaDataDatabase
from src.core.expressions.expression import ArrayLiteralExpr, DictLiteralExpr, Expression
from src.core.expressions.scope import ExecutionScope
from src.core.expressions.statement import BlockStmt
from src.core.struct.struct import Struct
from src.core.struct.struct_attribute import StructAttribute
from src.core.struct.struct_method import StructMethod
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime
from src.core.types.types import (
    Type,
    TypeStr,
    TypeBool,
    TypeDateTime,
    TypeColor,
    TypeArray,
    TypeDict,
    Int2,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt2,
    UInt4,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float8_E4M3,
    Float8_E5M2,
    Float16,
    BFloat16,
    Float32,
    Float64,
    UFloat8_E4M3,
    UFloat8_E5M2,
    UFloat16,
    UBFloat16,
    UFloat32,
    UFloat64,
)
from src.parser.expressions.type_parser import TmpType
from src.parser.nastruct_parser.parser_state import NaStructParserState
from src.parser.nastruct_parser.tmp_structs import TmpAttribute, TmpFunction, TmpStruct


def build_builtin_types() -> dict[str, Type]:
    """
    Builds the mapping of builtin type names to their Type instances.

    Returns:
        dict[str, Type]: Builtin type catalog keyed by source name.
    """

    return {
        "Str": TypeStr(),
        "Bool": TypeBool(),
        "DateTime": TypeDateTime(),
        "Color": TypeColor(),
        "Int2": Int2,
        "Int4": Int4,
        "Int8": Int8,
        "Int16": Int16,
        "Int32": Int32,
        "Int64": Int64,
        "UInt2": UInt2,
        "UInt4": UInt4,
        "UInt8": UInt8,
        "UInt16": UInt16,
        "UInt32": UInt32,
        "UInt64": UInt64,
        "Float8_E4M3": Float8_E4M3,
        "Float8_E5M2": Float8_E5M2,
        "Float16": Float16,
        "BFloat16": BFloat16,
        "Float32": Float32,
        "Float64": Float64,
        "UFloat8_E4M3": UFloat8_E4M3,
        "UFloat8_E5M2": UFloat8_E5M2,
        "UFloat16": UFloat16,
        "UBFloat16": UBFloat16,
        "UFloat32": UFloat32,
        "UFloat64": UFloat64,
    }


def build_type_globals() -> dict[str, Any]:
    """
    Builds the global symbol table used to evaluate expressions at runtime.

    Contains the home type constructors callable from defaults, scripts, and
    queries (e.g. 'DateTime.now()', 'Color(255, 255, 255)').

    Returns:
        dict[str, Any]: Global symbols keyed by name.
    """

    return {
        "DateTime": DateTime,
        "Color": Color,
    }


class TypeResolver:
    """
    Resolves TmpType trees into Type instances.

    Handles builtin types, home types, generics (Array, Dict), and struct
    types. Struct types may be resolved on demand through a provider
    callback, allowing forward references regardless of declaration order.
    """

    def __init__(self) -> None:
        """
        Initializes the TypeResolver instance.
        """

        self.builtin_types: dict[str, Type] = build_builtin_types()
        self.struct_types: dict[str, Struct] = {}

        # Optional callback resolving a struct name into its Struct object
        # (set by the SchemaResolver to support forward references)
        self.struct_provider: Optional[Callable[[str], Struct]] = None

    def register_struct_type(self, struct: Struct) -> None:
        """
        Makes a resolved struct usable as an attribute / parameter type.

        Args:
            struct (Struct): Resolved struct to register.
        """

        self.struct_types[struct.name] = struct

    def resolve(self, tmp_type: TmpType) -> Type:
        """
        Resolves a TmpType tree into a Type instance.

        Args:
            tmp_type (TmpType): Parsed type tree to resolve.

        Returns:
            Type: The resolved schema type.

        Raises:
            TypeError: On unknown types or invalid generic arity.
        """

        # Non generic type: builtin, home, or struct
        if not tmp_type.children:

            if tmp_type.name in self.builtin_types:
                return self.builtin_types[tmp_type.name]

            if tmp_type.name in self.struct_types:
                return cast(Type, self.struct_types[tmp_type.name])

            # Resolve the struct on demand (raises if truly unknown);
            # Struct duck types as Type through its check_value method
            if self.struct_provider is not None:
                return cast(Type, self.struct_provider(tmp_type.name))

            raise TypeError(f"Unknown type '{tmp_type.name}'")

        # Resolve every type argument first
        resolved_children: list[Type] = [
            self.resolve(tmp_type=child) for child in tmp_type.children
        ]

        # Array takes exactly one element type
        if tmp_type.name == "Array":
            if len(resolved_children) != 1:
                raise TypeError("Type 'Array' takes exactly one type argument")

            return TypeArray(value_type=resolved_children[0])

        # Dict takes exactly a key type and a value type (Dict[K, V])
        if tmp_type.name == "Dict":
            if len(resolved_children) != 2:
                raise TypeError("Type 'Dict' takes exactly two type arguments")

            return TypeDict(
                value_type=resolved_children[1],
                key_type=resolved_children[0]
            )

        raise TypeError(f"Unknown generic type '{tmp_type.name}'")


class SchemaResolver:
    """
    Resolves the intermediate representation of a parsing session into the
    database: structs, attributes, methods, and standalone functions.
    """

    def __init__(self, parser_state: NaStructParserState) -> None:
        """
        Initializes the SchemaResolver instance.

        Args:
            parser_state (NaStructParserState): Parsing session to resolve.
        """

        self.parser_state: NaStructParserState = parser_state
        self.database: NaDataDatabase = parser_state.database
        self.type_resolver: TypeResolver = TypeResolver()
        self.type_globals: dict[str, Any] = build_type_globals()

        # Resolved structs by name (also detects parent / type cycles)
        self.resolved_structs: dict[str, Struct] = {}
        self.in_progress: set[str] = set()

        # Let the type resolver resolve struct types on demand
        self.type_resolver.struct_provider = self.resolve_struct_by_name

    def resolve(self) -> NaDataDatabase:
        """
        Resolves every tmp struct and function into the database.

        Returns:
            NaDataDatabase: The populated database.
        """

        # Resolve structs (dependency order via recursive resolution)
        for struct_name in self.parser_state.tmp_structs:
            self.resolve_struct_by_name(name=struct_name)

        # Resolve standalone functions (all struct types are available now)
        for tmp_function in self.parser_state.tmp_functions.values():
            self.database.register_function(
                function=self.resolve_callable(tmp_function=tmp_function)
            )

        return self.database

    def resolve_struct_by_name(self, name: str) -> Struct:
        """
        Resolves a single struct by name, resolving its dependencies first.

        Args:
            name (str): Name of the struct to resolve.

        Returns:
            Struct: The resolved struct.

        Raises:
            TypeError: On unknown structs or inheritance cycles.
        """

        # Already resolved
        if name in self.resolved_structs:
            return self.resolved_structs[name]

        # Check the struct exists in the intermediate representation
        if name not in self.parser_state.tmp_structs:
            raise TypeError(f"Unknown struct or type '{name}'")

        # Detect inheritance and type cycles
        if name in self.in_progress:
            raise TypeError(f"Circular struct dependency involving '{name}'")

        self.in_progress.add(name)

        tmp_struct: TmpStruct = self.parser_state.tmp_structs[name]

        # Resolve the parent first (depth first)
        parent: Optional[Struct] = (
            self.resolve_struct_by_name(name=tmp_struct.struct_parent)
            if tmp_struct.struct_parent is not None
            else None
        )

        struct: Struct = Struct(name=name, parent=parent)

        # Register before attributes so self references (e.g. Array[Entity]
        # inside Entity) resolve to this very struct object
        self.resolved_structs[name] = struct
        self.type_resolver.register_struct_type(struct=struct)
        self.database.register_struct(struct=struct)

        # Resolve attributes and methods
        for tmp_attribute in tmp_struct.attributes.values():
            self.resolve_attribute(struct=struct, tmp_attribute=tmp_attribute)

        for tmp_function in tmp_struct.methods.values():
            struct.add_method(method=self.resolve_callable(tmp_function=tmp_function))

        self.in_progress.discard(name)

        return struct

    def resolve_attribute(
        self,
        struct: Struct,
        tmp_attribute: TmpAttribute
    ) -> None:
        """
        Resolves a single attribute declaration and adds it to the struct.

        Args:
            struct (Struct): Owning struct (already registered).
            tmp_attribute (TmpAttribute): Attribute to resolve.
        """

        attr_type: Type = self.type_resolver.resolve(tmp_type=tmp_attribute.attribute_type)

        default_value: Any = None
        has_default: bool = False

        # Evaluate the default expression under the declared type
        if tmp_attribute.attribute_default_value is not None:
            default_value = self.evaluate_default(
                expression=tmp_attribute.attribute_default_value,
                resolved_type=attr_type
            )
            has_default = True

        struct.add_attribute(attribute=StructAttribute(
            name=tmp_attribute.attribute_name,
            attr_type=attr_type,
            default_value=default_value,
            has_default=has_default
        ))

    def resolve_callable(self, tmp_function: TmpFunction) -> StructMethod:
        """
        Resolves a method or standalone function declaration.

        Args:
            tmp_function (TmpFunction): Callable to resolve.

        Returns:
            StructMethod: The resolved callable.
        """

        # Resolve parameter types and keep default expressions lazy
        params: dict[str, Type] = {}
        default_param_exprs: dict[str, Expression] = {}

        for parameter in tmp_function.fn_argument.values():
            params[parameter.attribute_name] = self.type_resolver.resolve(
                tmp_type=parameter.attribute_type
            )

            if parameter.attribute_default_value is not None:
                default_param_exprs[parameter.attribute_name] = (
                    parameter.attribute_default_value
                )

        return StructMethod(
            name=tmp_function.fn_name,
            params=params,
            return_type=self.type_resolver.resolve(
                tmp_type=tmp_function.fn_output_type
            ),
            default_param_exprs=default_param_exprs,
            body_block=BlockStmt(statements=tmp_function.fn_body)
        )

    def evaluate_default(
        self,
        expression: Expression,
        resolved_type: Type
    ) -> Any:
        """
        Evaluates a default value expression under its declared type.

        Empty collection literals get their declared type injected before
        evaluation, since their element type cannot be inferred.

        Args:
            expression (Any): Default value expression AST.
            resolved_type (Type): Declared attribute type.

        Returns:
            Any: The raw evaluated default value.

        Raises:
            TypeError: If the evaluated value violates the declared type.
        """

        # Inject the declared type into empty collection literals
        if (
            isinstance(expression, ArrayLiteralExpr)
            and expression.declared_type is None
            and isinstance(resolved_type, TypeArray)
        ):
            expression.declared_type = resolved_type

        if (
            isinstance(expression, DictLiteralExpr)
            and expression.declared_type is None
            and isinstance(resolved_type, TypeDict)
        ):
            expression.declared_type = resolved_type

        # Evaluate in a scope exposing the type constructors
        scope: ExecutionScope = ExecutionScope(globals_dict=self.type_globals)

        value: Any = expression.evaluate(scope=scope)

        raw_value: Any = value.get() if hasattr(value, "get") else value

        # Validate against the declared type (Struct is duck typed as Type)
        resolved_type.check_value(value=raw_value)

        return raw_value


def resolve_parser_state(parser_state: NaStructParserState) -> NaDataDatabase:
    """
    Resolves a full parsing session into its database.

    Args:
        parser_state (NaStructParserState): Parsing session to resolve.

    Returns:
        NaDataDatabase: The populated database.
    """

    return SchemaResolver(parser_state=parser_state).resolve()
