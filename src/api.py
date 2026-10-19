"""
High level API for using Nadata from other programs.

Exposes the Nadata facade: parse projects (schemas plus data files), open
persisted databases, and interact with instances, functions, and queries
using plain Python values.
"""

# Import Modules
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional, cast

from src.core.database.database import NaDataDatabase
from src.core.database.instance import StructInstance
from src.core.database.persistence import dump_database, load_database
from src.core.database.queryset import QuerySet
from src.core.expressions.expression import LiteralExpr
from src.core.expressions.scope import ExecutionScope
from src.core.struct.struct import Struct
from src.core.types.home_types.color import Color
from src.core.types.home_types.datetime import DateTime
from src.core.types.types import (
    Float64,
    Int64,
    Type,
    TypeArray,
    TypeBool,
    TypeDict,
    TypeStr,
)
from src.core.values.values import (
    TypeExternal,
    Value,
    ValueArray,
    ValueDict,
    ValueExternal,
)
from src.parser.nadata_loader.loader import (
    NaDataLoader,
    collect_polymorphic_instances,
)
from src.parser.nastruct_parser.nastruct_parser import NaStructParser


def _wrap_argument(value: Any) -> Value:
    """
    Wraps a plain Python value into a typed runtime Value.

    Supports existing Values, struct instances, native scalars, and lists /
    dicts of those (element and key types inferred from the first item).

    Args:
        value (Any): Plain Python value to wrap.

    Returns:
        Value: Typed runtime Value.

    Raises:
        TypeError: If the value type is not supported.
    """

    if isinstance(value, Value):
        return value

    if isinstance(value, StructInstance):
        return ValueExternal(payload=value)

    if isinstance(value, (bool, int, float, str, DateTime, Color)):
        return LiteralExpr(value=value).value

    if isinstance(value, list):
        wrapped_elements: list[Value] = [_wrap_argument(value=element) for element in value]

        element_type: Any = (
            wrapped_elements[0].value_type if wrapped_elements else TypeExternal()
        )

        array_type: TypeArray = TypeArray(value_type=cast(Type, element_type))

        return ValueArray(
            value_type=array_type,
            elements=[wrapped.get() for wrapped in wrapped_elements],
        )

    if isinstance(value, dict):
        wrapped_entries: dict[Any, Value] = {
            key: _wrap_argument(value=entry) for key, entry in value.items()
        }

        first_entry: Optional[Value] = (
            next(iter(wrapped_entries.values())) if wrapped_entries else None
        )

        first_key: Any = next(iter(wrapped_entries)) if wrapped_entries else None

        entry_type: Any = first_entry.value_type if first_entry is not None else TypeExternal()

        dict_type: TypeDict = TypeDict(
            value_type=cast(Type, entry_type),
            key_type=cast(Type, _infer_key_type(first_key=first_key)),
        )

        return ValueDict(
            value_type=dict_type,
            entries={key: wrapped.get() for key, wrapped in wrapped_entries.items()},
        )

    raise TypeError(f"Cannot wrap argument of type '{type(value).__name__}'")


def _infer_key_type(first_key: Any) -> Any:
    """
    Infers a schema key type from the first dict entry key.

    Args:
        first_key (Any): First key of the dict argument.

    Returns:
        Any: Inferred key schema type.
    """

    if isinstance(first_key, bool):
        return TypeBool()

    if isinstance(first_key, int):
        return Int64

    if isinstance(first_key, float):
        return Float64

    if isinstance(first_key, str):
        return TypeStr()

    return TypeExternal()


def _unwrap_value(val: Any) -> Any:
    """
    Unwraps a runtime Value into its raw Python payload.

    Args:
        val (Any): Value or raw payload to unwrap.

    Returns:
        Any: Raw underlying payload.
    """

    return val.get() if isinstance(val, Value) else val


class Nadata:
    """
    High level facade over a Nadata database.

    Loads projects (schema folders, single schema files, or persisted
    database files) and exposes instance access, function and method calls,
    polymorphic queries, and persistence using plain Python values.

    Attributes:
        database (NaDataDatabase): Underlying database engine.
        _globals (Optional[dict[str, Any]]): Lazy runtime globals.
        _scope (Optional[ExecutionScope]): Lazy execution scope.
    """

    def __init__(self, database: NaDataDatabase) -> None:
        """
        Initializes the Nadata facade over an existing database.

        Args:
            database (NaDataDatabase): Database to expose.
        """

        self.database: NaDataDatabase = database
        self._globals: Optional[dict[str, Any]] = None
        self._scope: Optional[ExecutionScope] = None

    # --- Construction ---

    @classmethod
    def load(cls, path: str) -> "Nadata":
        """
        Loads a Nadata source: a project folder, a schema file, or a
        persisted database file.

        Args:
            path (str): Path to load:
                - directory: parses every .nastruct (schemas) and .nadata
                  (data) file found recursively;
                - .nastruct file: parses a single schema file;
                - .ndb file: opens a persisted database (same as open);
                - .nadata file: not supported (data requires its schema).

        Returns:
            Nadata: The loaded facade.

        Raises:
            ValueError: If the path type is unsupported.
        """

        if os.path.isdir(path):
            return cls._load_project(folder_path=path)

        if path.endswith(".ndb"):
            return cls.open(storage_path=path)

        if path.endswith(".nastruct"):
            return cls._load_schema_file(file_path=path)

        if path.endswith(".nadata"):
            raise ValueError(
                f"Cannot load data file '{path}' alone: load the project "
                f"folder containing its schema instead"
            )

        raise ValueError(f"Unsupported Nadata path '{path}'")

    @classmethod
    def open(cls, storage_path: str) -> "Nadata":
        """
        Opens a persisted database file previously written by dump.

        Args:
            storage_path (str): Path to the .ndb database file.

        Returns:
            Nadata: The reopened facade.
        """

        return cls(database=load_database(storage_path=storage_path))

    @classmethod
    def _load_project(cls, folder_path: str) -> "Nadata":
        """
        Parses a project folder: schemas first, then data files.

        Args:
            folder_path (str): Folder containing .nastruct and .nadata files.

        Returns:
            Nadata: The loaded facade.
        """

        parser: NaStructParser = NaStructParser()
        database: NaDataDatabase = parser.parse_folder(folder_path=folder_path)

        if any(Path(folder_path).rglob("*.nadata")):
            loader: NaDataLoader = NaDataLoader(database=database)
            loader.load_folder(folder_path=folder_path)

        return cls(database=database)

    @classmethod
    def _load_schema_file(cls, file_path: str) -> "Nadata":
        """
        Parses a single .nastruct schema file.

        Args:
            file_path (str): Path to the schema file.

        Returns:
            Nadata: The loaded facade (no instances).
        """

        parser: NaStructParser = NaStructParser()

        schema_file = open(file_path, "r", encoding="utf-8")
        content: str = schema_file.read()
        schema_file.close()

        parser.parse_file(content=content)

        return cls(database=parser.resolve())

    # --- Introspection ---

    @property
    def struct_names(self) -> list[str]:
        """
        Returns the registered struct names.

        Returns:
            list[str]: Struct names in registration order.
        """

        return list(self.database.tables_by_struct_name)

    @property
    def function_names(self) -> list[str]:
        """
        Returns the registered standalone function names.

        Returns:
            list[str]: Function names.
        """

        return list(self.database.functions)

    @property
    def symbol_names(self) -> list[str]:
        """
        Returns the bound instance symbol names.

        Returns:
            list[str]: Symbol names.
        """

        return list(self.database.symbols)

    def summary(self) -> str:
        """
        Renders a human readable summary of the database contents.

        Returns:
            str: Multi line summary.
        """

        lines: list[str] = [f"Structs ({len(self.struct_names)}):"]

        for struct_name in self.struct_names:
            table = self.database.tables_by_struct_name[struct_name]
            struct: Struct = table.struct

            parent_note: str = (
                f", extends {struct.parent.name}" if struct.parent is not None else ""
            )

            lines.append(
                f"  {struct_name}{parent_note}: "
                f"{len(struct.get_all_attributes())} attributes, "
                f"{len(struct.methods)} methods, "
                f"{len(table.instances)} instances"
            )

        functions_note: str = (
            ", ".join(sorted(self.function_names)) if self.function_names else "(none)"
        )
        lines.append(f"Functions ({len(self.function_names)}): {functions_note}")

        symbols_note: str = (
            ", ".join(sorted(self.symbol_names)) if self.symbol_names else "(none)"
        )
        lines.append(f"Symbols ({len(self.symbol_names)}): {symbols_note}")

        return "\n".join(lines)

    def total_instance_count(self) -> int:
        """
        Counts every stored instance across all tables.

        Returns:
            int: Total instance count.
        """

        return sum(
            len(table.instances)
            for table in self.database.tables_by_struct_name.values()
        )

    # --- Instance access ---

    def get(self, symbol: str) -> StructInstance:
        """
        Returns the instance bound to a symbol name.

        Args:
            symbol (str): Symbol name (e.g. "tom").

        Returns:
            StructInstance: The bound instance.

        Raises:
            KeyError: If the symbol is unknown.
        """

        instance: Optional[StructInstance] = self.database.get_by_symbol(name=symbol)

        if instance is None:
            raise KeyError(f"Unknown symbol '{symbol}'")

        return instance

    def get_attribute(self, symbol: str, name: str) -> Any:
        """
        Returns an instance attribute as a plain Python value.

        Args:
            symbol (str): Symbol name of the instance.
            name (str): Attribute name.

        Returns:
            Any: Raw attribute payload.
        """

        return _unwrap_value(self.get(symbol=symbol).get_attribute(name=name))

    def set_attribute(self, symbol: str, name: str, value: Any) -> None:
        """
        Sets an instance attribute from a plain Python value.

        Args:
            symbol (str): Symbol name of the instance.
            name (str): Attribute name.
            value (Any): Plain Python value (validated against the schema).
        """

        self.get(symbol=symbol).set_attribute(name=name, value=_wrap_argument(value=value))

    def create_instance(
        self,
        struct_name: str,
        attributes: Optional[dict[str, Any]] = None,
        symbol: Optional[str] = None
    ) -> StructInstance:
        """
        Creates a new instance of a struct with validated plain Python values.

        Args:
            struct_name (str): Name of the struct to instantiate.
            attributes (Optional[dict[str, Any]], optional): Attribute values.
                Defaults to None.
            symbol (Optional[str], optional): Symbol to bind. Defaults to None.

        Returns:
            StructInstance: The created instance.

        Raises:
            KeyError: If the struct name is unknown.
        """

        struct: Struct = self._struct_or_raise(struct_name=struct_name)

        instance: StructInstance = self.database.create_instance(
            struct=struct,
            attributes=None,
            symbol=symbol
        )

        for attr_name, attr_value in (attributes or {}).items():
            instance.set_attribute(name=attr_name, value=_wrap_argument(value=attr_value))

        return instance

    # --- Calls and queries ---

    def call_function(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Calls a standalone schema function with plain Python arguments.

        Args:
            name (str): Function name.
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            Any: Raw function result.

        Raises:
            KeyError: If the function name is unknown.
        """

        self._ensure_globals()

        if name not in self.database.functions:
            raise KeyError(f"Unknown function '{name}'")

        function = self.database.functions[name]

        result: Any = function.execute(
            self_instance=None,
            args=[_wrap_argument(value=arg) for arg in args],
            kwargs={key: _wrap_argument(value=value) for key, value in kwargs.items()},
            globals_dict=self._globals,
        )

        return _unwrap_value(result)

    def call_method(self, symbol: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Calls a method on a symbol bound instance with plain Python arguments.

        Args:
            symbol (str): Symbol name of the receiver instance.
            name (str): Method name.
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            Any: Raw method result.

        Raises:
            KeyError: If the symbol or method name is unknown.
        """

        self._ensure_globals()

        instance: StructInstance = self.get(symbol=symbol)
        method = instance.struct.get_method(name=name)

        if method is None:
            raise KeyError(f"Struct '{instance.struct.name}' has no method '{name}'")

        result: Any = method.execute(
            self_instance=instance,
            args=[_wrap_argument(value=arg) for arg in args],
            kwargs={key: _wrap_argument(value=value) for key, value in kwargs.items()},
            globals_dict=self._globals,
        )

        return _unwrap_value(result)

    def query(self, struct_name: str) -> QuerySet:
        """
        Returns a query set over every instance of a struct, including all
        subclass instances (polymorphic).

        Args:
            struct_name (str): Name of the struct to query.

        Returns:
            QuerySet: Query set supporting chained where() / count() / ...

        Raises:
            KeyError: If the struct name is unknown.
        """

        self._ensure_globals()

        struct: Struct = self._struct_or_raise(struct_name=struct_name)

        return QuerySet(
            struct=struct,
            instances=collect_polymorphic_instances(
                struct=struct,
                database=self.database
            ),
            database=self.database,
            query_executor=self.database.query_executor
        )

    def count(self, struct_name: str) -> int:
        """
        Counts every instance of a struct, including subclass instances.

        Args:
            struct_name (str): Name of the struct to count.

        Returns:
            int: Polymorphic instance count.
        """

        return len(self.query(struct_name=struct_name).to_list())

    # --- Persistence ---

    def dump(self, storage_path: str) -> None:
        """
        Persists the database to a slotted page file.

        Args:
            storage_path (str): Target .ndb file path (must not exist).
        """

        dump_database(database=self.database, storage_path=storage_path)

    # --- Internals ---

    def _struct_or_raise(self, struct_name: str) -> Struct:
        """
        Looks up a struct by name.

        Args:
            struct_name (str): Struct name to look up.

        Returns:
            Struct: The registered struct.

        Raises:
            KeyError: If no struct is registered under the name.
        """

        table = self.database.tables_by_struct_name.get(struct_name)

        if table is None:
            raise KeyError(f"Unknown struct '{struct_name}'")

        return table.struct

    def _ensure_globals(self) -> None:
        """
        Builds the runtime globals and scope on first use.

        Built lazily so that instances created after construction (data
        loading, create_instance) are visible to calls and queries.
        """

        if self._globals is None:
            self._globals = NaDataLoader.build_runtime_globals(database=self.database)
            self._scope = ExecutionScope(globals_dict=self._globals)
