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
from src.core.expressions.builtins import builtin_type
from src.core.expressions.expression import (
    ArrayLiteralExpr,
    DictLiteralExpr,
    Expression,
    SetLiteralExpr,
)
from src.core.expressions.scope import ExecutionScope
from src.core.expressions.statement import (
    BlockStmt,
    BreakStmt,
    ContinueStmt,
    ForInStmt,
    ForStmt,
    IfStmt,
    Statement,
    VarDeclStmt,
    WhileStmt,
)
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
    TypeSet,
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
from src.parser.expressions.type_parser import TmpType, TypeParser
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
        "type": builtin_type,
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

        # Set takes exactly one element type
        if tmp_type.name == "Set":
            if len(resolved_children) != 1:
                raise TypeError("Type 'Set' takes exactly one type argument")

            return TypeSet(value_type=resolved_children[0])

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

        self.check_identifier_shadowing(tmp_struct=tmp_struct, struct=struct)

        self.in_progress.discard(name)

        return struct

    def check_identifier_shadowing(self, tmp_struct: TmpStruct, struct: Struct) -> None:
        """
        Rejects method parameters or locals that shadow struct attributes.

        Bare names in method bodies resolve to parameters, locals, and
        globals only; attributes are reached through explicit 'self.'. A
        parameter or local sharing a name with an attribute would silently
        take precedence over it, so it is a resolution error.

        Args:
            tmp_struct (TmpStruct): Intermediate struct with method bodies.
            struct (Struct): Resolved struct owning the attributes.

        Raises:
            SyntaxError: If a method parameter or local variable name
                collides with an attribute of the struct or its parents.
        """

        attribute_names: set[str] = set(struct.attributes.keys())

        parent: Optional[Struct] = struct.parent

        while parent is not None:
            attribute_names.update(parent.attributes.keys())
            parent = parent.parent

        for tmp_function in tmp_struct.methods.values():
            self._resolve_body_statement_types(
                statements=tmp_function.fn_body,
                fn_name=tmp_function.fn_name,
                struct_name=struct.name,
            )

            declared_names: list[str] = list(tmp_function.fn_argument.keys())

            for statement in tmp_function.fn_body:
                self._collect_declared_names(statement=statement, names=declared_names)

            for declared_name in declared_names:
                if declared_name in attribute_names:
                    raise SyntaxError(
                        f"'{declared_name}' in method '{tmp_function.fn_name}' "
                        f"of struct '{struct.name}' shadows an attribute: "
                        f"access attributes through 'self'"
                    )

            # Loop variables must be fresh in their lexical scope
            self._check_loop_variable_freshness(
                statements=tmp_function.fn_body,
                visible=set(tmp_function.fn_argument.keys()),
                fn_name=tmp_function.fn_name,
                struct_name=struct.name,
            )

    def _resolve_body_statement_types(
        self,
        statements: list[Statement],
        fn_name: str,
        struct_name: str,
    ) -> None:
        """
        Resolves the raw type annotations inside a method body statement tree.

        Covers typed local declarations and loop variables (recursing into
        nested blocks, branches, and loop bodies).

        Args:
            statements (list[Statement]): Statement tree to resolve in place.
            fn_name (str): Method name, for error messages.
            struct_name (str): Struct name, for error messages.

        Raises:
            SyntaxError: If a type annotation cannot be resolved.
        """

        for statement in statements:
            if isinstance(statement, VarDeclStmt) and statement.var_type is None:
                if statement.var_type_name is not None:
                    statement.var_type = self._resolve_annotation(
                        type_name=statement.var_type_name,
                        fn_name=fn_name,
                        struct_name=struct_name,
                    )

            elif isinstance(statement, ForInStmt) and statement.var_type is None:
                if statement.var_type_name is not None:
                    statement.var_type = self._resolve_annotation(
                        type_name=statement.var_type_name,
                        fn_name=fn_name,
                        struct_name=struct_name,
                    )

                self._resolve_body_statement_types(
                    statements=statement.body.statements,
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

            elif isinstance(statement, ForStmt):
                self._resolve_body_statement_types(
                    statements=statement.init_statements,
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

                self._resolve_body_statement_types(
                    statements=statement.body.statements,
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

            elif isinstance(statement, WhileStmt):
                self._resolve_body_statement_types(
                    statements=statement.body.statements,
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

            elif isinstance(statement, IfStmt):
                self._resolve_body_statement_types(
                    statements=statement.then_block.statements,
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

                if statement.else_block is not None:
                    self._resolve_body_statement_types(
                        statements=statement.else_block.statements,
                        fn_name=fn_name,
                        struct_name=struct_name,
                    )

            elif isinstance(statement, BlockStmt):
                self._resolve_body_statement_types(
                    statements=statement.statements,
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

    def _resolve_annotation(self, type_name: str, fn_name: str, struct_name: str) -> Type:
        """
        Resolves a raw type annotation from a method body.

        Args:
            type_name (str): Raw annotation text.
            fn_name (str): Method name, for error messages.
            struct_name (str): Struct name, for error messages.

        Returns:
            Type: The resolved schema type.

        Raises:
            SyntaxError: If the annotation cannot be parsed or resolved.
        """

        try:
            tmp_type: TmpType = TypeParser.parse_text(text=type_name)

            return self.type_resolver.resolve(tmp_type=tmp_type)

        except SyntaxError as error:
            raise SyntaxError(
                f"Unknown type '{type_name}' for local in method '{fn_name}' "
                f"of struct '{struct_name}': {error}"
            )

    def _check_loop_variable_freshness(
        self,
        statements: list[Statement],
        visible: set[str],
        fn_name: str,
        struct_name: str,
    ) -> None:
        """
        Rejects loop variables that shadow a name visible at the loop.

        The walk is flow sensitive: only parameters and locals declared
        lexically before the loop are in scope. Loop variables themselves go
        out of scope after their loop, so sequential loops may reuse a name.

        Args:
            statements (list[Statement]): Statement sequence to walk.
            visible (set[str]): Names visible at the current position.
            fn_name (str): Method name, for error messages.
            struct_name (str): Struct name, for error messages.

        Raises:
            SyntaxError: If a loop variable shadows a visible name.
        """

        for statement in statements:
            if isinstance(statement, VarDeclStmt):
                visible.add(statement.name)

            elif isinstance(statement, ForInStmt):
                if statement.var_name in visible:
                    raise SyntaxError(
                        f"Loop variable '{statement.var_name}' in method "
                        f"'{fn_name}' of struct '{struct_name}' shadows an "
                        f"existing variable in scope"
                    )

                self._check_loop_variable_freshness(
                    statements=statement.body.statements,
                    visible=visible | {statement.var_name},
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

            elif isinstance(statement, ForStmt):
                init_names: set[str] = {
                    init.name
                    for init in statement.init_statements
                    if isinstance(init, VarDeclStmt)
                }

                for init_name in init_names:
                    if init_name in visible:
                        raise SyntaxError(
                            f"Loop variable '{init_name}' in method "
                            f"'{fn_name}' of struct '{struct_name}' shadows an "
                            f"existing variable in scope"
                        )

                self._check_loop_variable_freshness(
                    statements=statement.body.statements,
                    visible=visible | init_names,
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

            elif isinstance(statement, WhileStmt):
                self._check_loop_variable_freshness(
                    statements=statement.body.statements,
                    visible=set(visible),
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

            elif isinstance(statement, IfStmt):
                self._check_loop_variable_freshness(
                    statements=statement.then_block.statements,
                    visible=set(visible),
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

                if statement.else_block is not None:
                    self._check_loop_variable_freshness(
                        statements=statement.else_block.statements,
                        visible=set(visible),
                        fn_name=fn_name,
                        struct_name=struct_name,
                    )

            elif isinstance(statement, BlockStmt):
                self._check_loop_variable_freshness(
                    statements=statement.statements,
                    visible=set(visible),
                    fn_name=fn_name,
                    struct_name=struct_name,
                )

    def _collect_declared_names(self, statement: Statement, names: list[str]) -> None:
        """
        Appends the names of variables declared by a statement tree.

        Args:
            statement (Statement): Statement to inspect (recursed into
                blocks and branches).
            names (list[str]): Accumulator for declared variable names.
        """

        if isinstance(statement, VarDeclStmt):
            names.append(statement.name)

            return

        if isinstance(statement, ForInStmt):
            names.append(statement.var_name)

            for nested in statement.body.statements:
                self._collect_declared_names(statement=nested, names=names)

            return

        if isinstance(statement, ForStmt):
            for init in statement.init_statements:
                self._collect_declared_names(statement=init, names=names)

            for nested in statement.body.statements:
                self._collect_declared_names(statement=nested, names=names)

            return

        if isinstance(statement, WhileStmt):
            for nested in statement.body.statements:
                self._collect_declared_names(statement=nested, names=names)

            return

        if isinstance(statement, BlockStmt):
            for nested in statement.statements:
                self._collect_declared_names(statement=nested, names=names)

            return

        if isinstance(statement, IfStmt):
            for nested in statement.then_block.statements:
                self._collect_declared_names(statement=nested, names=names)

            if statement.else_block is not None:
                for nested in statement.else_block.statements:
                    self._collect_declared_names(statement=nested, names=names)

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

        # A set declared with '{}' reuses the (otherwise dict) empty braces
        if (
            isinstance(expression, DictLiteralExpr)
            and not expression.entries
            and isinstance(resolved_type, TypeSet)
        ):
            expression = SetLiteralExpr(elements=[], declared_type=resolved_type)

        if (
            isinstance(expression, SetLiteralExpr)
            and expression.declared_type is None
            and isinstance(resolved_type, TypeSet)
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
