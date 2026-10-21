"""
Data loader for Nadata .nadata files.

Parses .nadata source into statements, creates and links struct instances
(two phase, supporting forward and circular references), then executes the
remaining script statements (function calls, typed declarations, queries).
"""

# Import Modules
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, cast

from src.core.struct.struct import Struct
from src.core.values.values import Value, ValueExternal
from src.core.expressions.scope import DEFAULT_LOOP_LIMIT, LOOP_FUEL_KEY, ExecutionScope, LoopFuel
from src.core.expressions.statement import (
    AssignStmt,
    BlockStmt,
    ForInStmt,
    ForStmt,
    IfStmt,
    Statement,
    VarDeclStmt,
    WhileStmt,
)
from src.core.database.database import NaDataDatabase
from src.core.database.instance import StructInstance
from src.core.expressions.builtins import (
    SchemaClassesFunction,
    SchemaFunctionsFunction,
    SetLoopLimitFunction,
)
from src.core.database.queryset import QuerySet
from src.core.expressions.expression import CallExpr, Expression, IdentifierExpr
from src.parser.expressions.type_parser import TmpType, TypeParser
from src.parser.utils.comments_remove import CommentsRemover
from src.parser.nastruct_parser.resolver import TypeResolver, build_type_globals
from src.parser.expressions.statement_parser import StatementParser


class StructRuntimeBinding:
    """
    Runtime binding of a struct name to its database table.

    Placed in loader globals under the struct name. Callable as a constructor
    creating new instances, and exposing query entry points such as where().

    Attributes:
        struct (Struct): Bound struct schema.
        database (NaDataDatabase): Owning database used for instance creation.
    """

    def __init__(self, struct: Struct, database: NaDataDatabase) -> None:
        """
        Initializes the StructRuntimeBinding instance.

        Args:
            struct (Struct): Bound struct schema.
            database (NaDataDatabase): Owning database used for instance creation.
        """

        self.struct: Struct = struct
        self.database: NaDataDatabase = database

    def __call__(
        self,
        *args: Value,
        **kwargs: Value
    ) -> StructInstance:
        """
        Creates a new instance, binding positional args to attributes in
        declaration order and keyword args by name.

        Args:
            *args (Value): Positional attribute values in declaration order.
            **kwargs (Value): Named attribute values.

        Returns:
            StructInstance: The created (unregistered) instance.

        Raises:
            TypeError: If more positional args are given than declared attributes.
        """

        attribute_names: list[str] = list(self.struct.get_all_attributes().keys())

        if len(args) > len(attribute_names):
            raise TypeError(
                f"Struct '{self.struct.name}' takes at most {len(attribute_names)} "
                f"positional attributes but {len(args)} given"
            )

        instance: StructInstance = self.database.create_instance(
            struct=self.struct,
            attributes=None
        )

        # Bind positional arguments in declaration order
        for idx, arg_value in enumerate(args):
            instance.set_attribute(name=attribute_names[idx], value=arg_value)

        # Bind keyword arguments by name
        for kw_name, kw_value in kwargs.items():
            instance.set_attribute(name=kw_name, value=kw_value)

        return instance

    def order_by(
        self,
        attribute: Value,
        descending: Optional[Value] = None,
    ) -> QuerySet:
        """
        Returns all instances of the struct sorted by an attribute.

        Args:
            attribute (Value): Wrapped attribute name to sort by.
            descending (Value, optional): Wrapped descending flag.
                Defaults to ascending.

        Returns:
            QuerySet: Sorted query set over the polymorphic instance list.
        """

        return self._base_queryset().order_by(
            attribute=attribute,
            descending=bool(descending.get()) if descending is not None else False,
        )

    def sum_of(self, attribute: Value) -> Any:
        """
        Sums a numeric attribute over all instances of the struct.

        Args:
            attribute (Value): Wrapped attribute name.

        Returns:
            Any: The sum payload.
        """

        return self._base_queryset().sum_of(attribute=attribute)

    def min_of(self, attribute: Value) -> Any:
        """
        Returns the minimum of a numeric attribute over all instances.

        Args:
            attribute (Value): Wrapped attribute name.

        Returns:
            Any: The minimum payload.
        """

        return self._base_queryset().min_of(attribute=attribute)

    def max_of(self, attribute: Value) -> Any:
        """
        Returns the maximum of a numeric attribute over all instances.

        Args:
            attribute (Value): Wrapped attribute name.

        Returns:
            Any: The maximum payload.
        """

        return self._base_queryset().max_of(attribute=attribute)

    def average_of(self, attribute: Value) -> Any:
        """
        Returns the mean of a numeric attribute over all instances.

        Args:
            attribute (Value): Wrapped attribute name.

        Returns:
            Any: The mean payload.
        """

        return self._base_queryset().average_of(attribute=attribute)

    def _base_queryset(self) -> QuerySet:
        """
        Builds the unfiltered polymorphic query set for this struct.

        Args:
            (none)

        Returns:
            QuerySet: Query set over every instance (subclasses included).
        """

        return QuerySet(
            struct=self.struct,
            instances=collect_polymorphic_instances(
                struct=self.struct,
                database=self.database
            ),
            database=self.database,
            query_executor=self.database.query_executor,
        )

    def where(self, predicate: Value) -> QuerySet:
        """
        Filters all instances of the struct (including subclasses) by a predicate.

        Args:
            predicate (Value): ValueExternal wrapping a callable candidate predicate.

        Returns:
            QuerySet: Filtered query set over the polymorphic instance list.

        Raises:
            TypeError: If the predicate does not resolve to a callable.
        """

        callable_predicate: Any = (
            predicate.get() if isinstance(predicate, Value) else predicate
        )

        if not callable(callable_predicate):
            raise TypeError(
                f"'.where' on struct '{self.struct.name}' expects a callable predicate"
            )

        return QuerySet(
            struct=self.struct,
            instances=collect_polymorphic_instances(
                struct=self.struct,
                database=self.database
            ),
            database=self.database,
            query_executor=self.database.query_executor
        ).where(predicate=callable_predicate)

    def __repr__(self) -> str:
        """
        Returns string representation of the binding.

        Returns:
            str: Binding representation.
        """

        return f"StructRuntimeBinding({self.struct.name})"


def collect_polymorphic_instances(
    struct: Struct,
    database: NaDataDatabase
) -> list[StructInstance]:
    """
    Collects every instance of a struct and of all its subclasses.

    Args:
        struct (Struct): Struct whose instance hierarchy is collected.
        database (NaDataDatabase): Database to search for instance tables.

    Returns:
        list[StructInstance]: Instances from all matching tables.
    """

    instances: list[StructInstance] = []

    for table in database.tables_by_struct_name.values():
        if table.struct is struct or table.struct.is_subclass_of(other=struct):
            instances.extend(table.all())

    return instances


def format_runtime_value(val: Any) -> str:
    """
    Formats a runtime Value for the print builtin.

    Query results render as instance lists, other payloads render raw.

    Args:
        val (Any): Value or raw payload to format.

    Returns:
        str: Human readable representation.
    """

    raw: Any = val.get() if isinstance(val, Value) else val

    # Duck-typed query results render as instance lists
    if hasattr(raw, "to_list") and hasattr(raw, "where"):
        return "[" + ", ".join(repr(inst) for inst in raw.to_list()) + "]"

    # External wrappers render through their own repr (e.g. type handles)
    if isinstance(val, ValueExternal):
        return repr(val)

    return str(raw)


def print_runtime_values(*args: Value) -> None:
    """
    Print builtin formatting each argument Value.

    Args:
        *args (Value): Argument Values to print space separated.
    """

    print(" ".join(format_runtime_value(val=val) for val in args))


class NaDataLoader:
    """
    Loads .nadata data files into an existing resolved NaDataDatabase.

    Instance declarations are detected as assignments whose right hand side is
    a call on a registered struct name. Declaration and population are split
    in two phases so that forward and circular references between instances
    resolve correctly. Remaining statements execute as a script sharing the
    instance symbol scope.

    Attributes:
        database (NaDataDatabase): Target database the data is loaded into.
        declarations (list[tuple[str, str, AssignStmt]]): Pending instance
            declarations as (struct_name, symbol, statement) tuples.
        script_statements (list[Statement]): Non-declaration statements.
        instance_bindings (dict[str, ValueExternal]): Symbol to instance bindings.
        globals_dict (dict[str, Any]): Global symbols for script execution.
        scope (Optional[ExecutionScope]): Shared execution scope for all passes.
        type_resolver (TypeResolver): Resolver for declared variable types.
        _type_cache (dict[str, Any]): Resolved variable type cache by annotation.
    """

    def __init__(self, database: NaDataDatabase) -> None:
        """
        Initializes the NaDataLoader instance.

        Args:
            database (NaDataDatabase): Target database the data is loaded into.
        """

        self.database: NaDataDatabase = database
        self.declarations: list[tuple[str, str, AssignStmt]] = []
        self.script_statements: list[Statement] = []
        self.instance_bindings: dict[str, ValueExternal] = {}
        self.globals_dict: dict[str, Any] = {}
        self.scope: ExecutionScope = ExecutionScope(globals_dict={})
        self.type_resolver: TypeResolver = TypeResolver()
        self.type_resolver.struct_provider = self._resolve_struct_type
        self._type_cache: dict[str, Any] = {}

    # --- Public API ---

    def load_file(self, content: str) -> None:
        """
        Parses and loads a single .nadata source content.

        Args:
            content (str): Raw .nadata source text.
        """

        self._collect_statements(statements=self._parse(content=content))
        self._run_passes()

    def load_folder(self, folder_path: str) -> None:
        """
        Loads every .nadata file found recursively in a folder with a shared
        symbol scope.

        Args:
            folder_path (str): Folder containing .nadata files.
        """

        data_files: list[Path] = sorted(Path(folder_path).rglob("*.nadata"))

        for data_file in data_files:
            self._collect_statements(
                statements=self._parse(content=data_file.read_text(encoding="utf-8"))
            )

        self._run_passes()

    # --- Pass 0: parsing and classification ---

    def parse_statements(self, content: str) -> list[Statement]:
        """
        Parses .nadata source content into statements without executing it.

        Args:
            content (str): Raw .nadata source text.

        Returns:
            list[Statement]: Parsed statements in source order.
        """

        return self._parse(content=content)

    def _parse(self, content: str) -> list[Statement]:
        """
        Strips comments and parses the source into statements.

        Args:
            content (str): Raw .nadata source text.

        Returns:
            list[Statement]: Parsed statements in source order.
        """

        clean_content: str = CommentsRemover.remove_all_comments(content=content)

        return StatementParser.parse_body(text=clean_content)

    def _collect_statements(self, statements: list[Statement]) -> None:
        """
        Splits statements into instance declarations and script statements.

        Args:
            statements (list[Statement]): Parsed statements to classify.
        """

        for stmt in statements:
            struct_name: Optional[str] = self._instance_struct_name(stmt=stmt)

            if struct_name is not None and isinstance(stmt, AssignStmt):
                self.declarations.append(
                    (struct_name, stmt.target_name, stmt)
                )

            else:
                self.script_statements.append(stmt)

    def _instance_struct_name(self, stmt: Statement) -> Optional[str]:
        """
        Returns the struct name if the statement declares an instance, else None.

        Args:
            stmt (Statement): Statement to inspect.

        Returns:
            Optional[str]: Struct name of the declared instance, or None.
        """

        if not isinstance(stmt, AssignStmt):
            return None

        if not isinstance(stmt.value_expr, CallExpr):
            return None

        callee: Expression = stmt.value_expr.callee

        if not isinstance(callee, IdentifierExpr):
            return None

        if callee.name not in self.database.tables_by_struct_name:
            return None

        return callee.name

    # --- Pass 1: instance creation ---

    def _declare_instances(self) -> None:
        """
        Creates every declared instance empty and binds its symbol.

        All instances exist before any attribute is populated, so forward and
        circular references between symbols evaluate correctly in pass 2.
        """

        for _, symbol, _ in self.declarations:
            struct_name: str = self._symbol_struct_name(symbol=symbol)
            struct: Struct = self.database.tables_by_struct_name[struct_name].struct

            instance: StructInstance = self.database.create_instance(
                struct=struct,
                attributes=None,
                symbol=symbol
            )

            self.instance_bindings[symbol] = ValueExternal(payload=instance)

    def _symbol_struct_name(self, symbol: str) -> str:
        """
        Finds the struct name declared for a symbol.

        Args:
            symbol (str): Instance symbol name.

        Returns:
            str: Declared struct name.

        Raises:
            ValueError: If the symbol is declared twice with different structs.
        """

        for struct_name, declared_symbol, _ in self.declarations:
            if declared_symbol == symbol:
                return struct_name

        raise ValueError(f"Symbol '{symbol}' has no pending declaration")

    # --- Pass 2: attribute population ---

    def _populate_instances(self) -> None:
        """
        Evaluates every declaration arguments and sets the instance attributes.

        Argument evaluation happens in the shared scope where all instance
        symbols already exist, enabling forward and circular references.
        """

        for _, symbol, stmt in self.declarations:
            instance: StructInstance = self.instance_bindings[symbol].get()
            call: CallExpr = cast(CallExpr, stmt.value_expr)

            attribute_values: dict[str, Value] = {}

            # Keyword arguments bind by name
            for kw_name, kw_expr in call.kwargs.items():
                attribute_values[kw_name] = kw_expr.evaluate(scope=self.scope)

            # Positional arguments bind to attributes in declaration order
            attribute_names: list[str] = list(instance.struct.get_all_attributes().keys())

            if len(call.args) > len(attribute_names):
                raise TypeError(
                    f"Instance '{symbol}' of '{instance.struct.name}' takes at most "
                    f"{len(attribute_names)} positional attributes but {len(call.args)} given"
                )

            for idx, arg_expr in enumerate(call.args):
                attribute_values[attribute_names[idx]] = arg_expr.evaluate(
                    scope=self.scope
                )

            for attr_name, attr_value in attribute_values.items():
                instance.set_attribute(name=attr_name, value=attr_value)

    # --- Pass 3: script execution ---

    def _execute_scripts(self) -> None:
        """
        Executes the remaining script statements in the shared scope.
        """

        for stmt in self.script_statements:
            self._resolve_script_statement_types(statement=stmt)

            if isinstance(stmt, VarDeclStmt):
                self._execute_var_decl(stmt=stmt)

            elif isinstance(stmt, AssignStmt):
                self._execute_assign(stmt=stmt)

            else:
                stmt.execute(scope=self.scope)

    def _resolve_script_statement_types(self, statement: Statement) -> None:
        """
        Resolves raw type annotations inside a script statement tree.

        Covers typed local declarations and loop variables (recursing into
        nested blocks, branches, and loop bodies). Resolution is idempotent:
        already resolved types are kept.

        Args:
            statement (Statement): Statement tree to resolve in place.
        """

        if isinstance(statement, VarDeclStmt) and statement.var_type is None:
            if statement.var_type_name is not None:
                statement.var_type = self._resolve_var_type(
                    type_name=statement.var_type_name
                )

            return

        if isinstance(statement, ForInStmt):
            if statement.var_type is None and statement.var_type_name is not None:
                statement.var_type = self._resolve_var_type(
                    type_name=statement.var_type_name
                )

            self._resolve_script_statement_types_list(
                statements=statement.body.statements
            )

            return

        if isinstance(statement, ForStmt):
            self._resolve_script_statement_types_list(
                statements=statement.init_statements
            )

            self._resolve_script_statement_types_list(
                statements=statement.body.statements
            )

            return

        if isinstance(statement, WhileStmt):
            self._resolve_script_statement_types_list(
                statements=statement.body.statements
            )

            return

        if isinstance(statement, IfStmt):
            self._resolve_script_statement_types_list(
                statements=statement.then_block.statements
            )

            if statement.else_block is not None:
                self._resolve_script_statement_types_list(
                    statements=statement.else_block.statements
                )

            return

        if isinstance(statement, BlockStmt):
            self._resolve_script_statement_types_list(
                statements=statement.statements
            )

    def _resolve_script_statement_types_list(self, statements: list[Statement]) -> None:
        """
        Resolves raw type annotations over a statement list.

        Args:
            statements (list[Statement]): Statements to resolve in place.
        """

        for statement in statements:
            self._resolve_script_statement_types(statement=statement)

    def _execute_var_decl(self, stmt: VarDeclStmt) -> None:
        """
        Resolves the declaration type annotation then executes the statement.

        Args:
            stmt (VarDeclStmt): Typed declaration statement.
        """

        if stmt.var_type_name is not None:
            stmt.var_type = self._resolve_var_type(type_name=stmt.var_type_name)

        stmt.execute(scope=self.scope)

    def _execute_assign(self, stmt: AssignStmt) -> None:
        """
        Executes an assignment, defining the name if it does not exist yet.

        Args:
            stmt (AssignStmt): Assignment statement.
        """

        value: Value = stmt.value_expr.evaluate(scope=self.scope)

        try:
            self.scope.get(name=stmt.target_name)

            # Existing name: reassign through the scope hierarchy
            self.scope.assign(name=stmt.target_name, value=value)

        except NameError:
            self.scope.define(name=stmt.target_name, value=value)

    # --- Helpers ---

    def _resolve_var_type(self, type_name: str) -> Any:
        """
        Resolves a raw type annotation into a schema Type.

        Args:
            type_name (str): Raw type annotation (e.g. 'Array[Human]').

        Returns:
            Any: Resolved schema Type instance.
        """

        if type_name in self._type_cache:
            return self._type_cache[type_name]

        tmp_type: TmpType = TypeParser.parse_text(text=type_name)

        resolved: Any = self.type_resolver.resolve(tmp_type=tmp_type)

        self._type_cache[type_name] = resolved

        return resolved

    def _resolve_struct_type(self, name: str) -> Struct:
        """
        Looks up a struct by name for the type resolver.

        Args:
            name (str): Struct name to look up.

        Returns:
            Struct: The registered struct.

        Raises:
            KeyError: If no struct is registered under the name.
        """

        return self.database.tables_by_struct_name[name].struct

    @staticmethod
    def build_runtime_globals(database: NaDataDatabase) -> dict[str, Any]:
        """
        Builds the global symbol table for evaluation.

        Contains home types, struct runtime bindings, standalone functions,
        instance symbol bindings, the print builtin, and the database-bound
        schema introspection functions.

        Args:
            database (NaDataDatabase): Database whose symbols are exposed.

        Returns:
            dict[str, Any]: Global symbol table.
        """

        globals_dict: dict[str, Any] = build_type_globals()

        globals_dict["print"] = print_runtime_values
        globals_dict["schema_classes"] = SchemaClassesFunction(database=database)
        globals_dict["schema_functions"] = SchemaFunctionsFunction(database=database)
        globals_dict["set_loop_limit"] = SetLoopLimitFunction(database=database)

        # Shared loop fuel, seeded from the persisted settings
        globals_dict[LOOP_FUEL_KEY] = LoopFuel(
            limit=int(
                database.settings.get("loop_limit", DEFAULT_LOOP_LIMIT)
            )
        )

        for struct_name, table in database.tables_by_struct_name.items():
            globals_dict[struct_name] = StructRuntimeBinding(
                struct=table.struct,
                database=database
            )

        for function_name, function in database.functions.items():
            globals_dict[function_name] = function

        for symbol_name, entity_id in database.symbols.items():
            instance: Optional[StructInstance] = database.get_by_symbol(name=symbol_name)

            if instance is not None:
                globals_dict[symbol_name] = ValueExternal(payload=instance)

        return globals_dict

    # --- Orchestration ---

    def _run_passes(self) -> None:
        """
        Runs the declare, populate, and execute passes in order.

        Variables from a previous load (e.g. an earlier load_file call in a
        console session) are carried into the fresh scope so that script
        locals persist across loads.
        """

        self._declare_instances()

        previous_variables: dict[str, Value] = dict(self.scope.variables)

        self.globals_dict = self.build_runtime_globals(database=self.database)

        self.scope = ExecutionScope(globals_dict=self.globals_dict)

        self.scope.variables.update(previous_variables)

        self._populate_instances()
        self._execute_scripts()
