"""
High level API for using Nadata from other programs.

Exposes the Nadata facade: parse projects (schemas plus data files), open
persisted databases, and interact with instances, functions, and queries
using plain Python values.
"""

# Import Modules
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any, Optional, cast

from nadata.core.database.database import NaDataDatabase
from nadata.core.database.instance import StructInstance
from nadata.core.database.persistence import (
    append_database_changes,
    compact_database,
    dump_database,
    load_database,
)
from nadata.core.database.persistence.dumper import flush_dirty_instances
from nadata.session import (
    MANUAL,
    ScriptFile,
    build_prelude,
    folder_is_session,
    load_folder_session,
    parse_script_file,
)
from nadata.core.database.queryset import QuerySet
from nadata.core.expressions.expression import LiteralExpr
from nadata.core.expressions.scope import ExecutionScope
from nadata.core.struct.struct import Struct
from nadata.core.types.home_types.color import Color
from nadata.core.types.home_types.datetime import DateTime
from nadata.core.types.types import (
    Float64,
    Int64,
    Type,
    TypeArray,
    TypeBool,
    TypeDict,
    TypeStr,
)
from nadata.core.values.values import (
    TypeExternal,
    Value,
    ValueArray,
    ValueDict,
    ValueExternal,
)
from nadata.parser.nadata_loader.loader import (
    NaDataLoader,
    collect_polymorphic_instances,
)
from nadata.parser.nastruct_parser.imports import resolve_schema_entry
from nadata.parser.nastruct_parser.nastruct_parser import NaStructParser


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
        self.session_folder: Optional[str] = None
        self.session_db: Optional[str] = None

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

        Folders with directive-marked scripts (#init/#auto/#manual) or an
        existing nadata.ndb world use session semantics instead: the
        world opens incrementally, #init runs once, #auto runs, and
        #manual scripts are left for explicit invocation.

        Args:
            folder_path (str): Folder containing .nastruct and .nadata files.

        Returns:
            Nadata: The loaded facade.
        """

        if folder_is_session(folder=Path(folder_path)):
            return cls.open_folder(folder_path=folder_path)

        parser: NaStructParser = NaStructParser()
        database: NaDataDatabase = parser.parse_folder(folder_path=folder_path)

        if any(Path(folder_path).rglob("*.nadata")):
            loader: NaDataLoader = NaDataLoader(database=database)
            loader.load_folder(folder_path=folder_path)

        return cls(database=database)

    @classmethod
    def open_folder(
        cls,
        folder_path: str,
        db_path: Optional[str] = None,
        run_auto: bool = True,
        force_init: bool = False,
    ) -> "Nadata":
        """
        Runs the folder session pipeline and binds the session context.

        Args:
            folder_path (str): Project folder.
            db_path (Optional[str], optional): World database file
                (defaults to '<folder>/nadata.ndb'). Defaults to None.
            run_auto (bool, optional): Whether to run #auto scripts.
                Defaults to True.
            force_init (bool, optional): Re-run #init scripts. Defaults
                to False.

        Returns:
            Nadata: The facade over the world database, with the session
                folder and db recorded for console commands.
        """

        database: NaDataDatabase
        result: Any

        database, result = load_folder_session(
            folder_path=folder_path,
            db_path=db_path,
            run_auto=run_auto,
            force_init=force_init,
        )

        app: Nadata = cls(database=database)

        app.session_folder = str(Path(folder_path))
        app.session_db = result.db_path

        return app

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

        # Imports resolve before parsing: dependency order, deduped
        for _path, content in resolve_schema_entry(entry=Path(file_path)):
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

        if attributes:
            wrapped: dict[str, Value] = {
                name: _wrap_argument(value=value) for name, value in attributes.items()
            }

            instance.set_creation_attributes(values=wrapped)

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

    def save_changes(self, storage_path: str) -> int:
        """
        Appends only new or changed data to an existing database file.

        The current state is diffed against the file contents: new structs,
        new or changed instances, and new symbol bindings are appended as a
        delta that the loader replays over the earlier snapshot. Calling
        this repeatedly on the same file is safe; unchanged data produces
        no records.

        Args:
            storage_path (str): Path to the existing .ndb file to extend.

        Returns:
            int: Number of delta documents appended.

        Raises:
            ValueError: If the file does not exist or is not a Nadata dump.
        """

        written: int = append_database_changes(
            database=self.database,
            storage_path=storage_path,
        )

        # The appended delta may have introduced new symbols or structs
        self._globals = None

        return written

    def set_loop_limit(self, limit: int) -> None:
        """
        Sets the maximum number of iterations allowed per loop.

        The limit is recorded in the database settings, so the next dump or
        incremental save persists it, and future executions read it from
        there. The default fuel limit is 10,000 iterations per loop; loops
        exceeding it raise LoopLimitError. The new limit applies to
        subsequent executions (call sites rebuild their scope lazily).

        Args:
            limit (int): New per-loop iteration limit (at least 1).

        Raises:
            TypeError: If the limit is not an integer.
            ValueError: If the limit is below 1.
        """

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(f"Loop limit must be an integer, got '{type(limit).__name__}'")

        if limit < 1:
            raise ValueError(f"Loop limit must be at least 1, got {limit}")

        self.database.settings["loop_limit"] = limit

        # Rebuild globals on next use: the fuel object is seeded from settings
        self._globals = None

    def compact(self, output_path: Optional[str] = None) -> tuple[int, int]:
        """
        Rewrites a database file into a minimal snapshot of the current state.

        Incremental saves append deltas that accumulate as replay history;
        compaction rewrites the file so it holds only live documents. The
        output is written to a temporary file and atomically renamed, so a
        crash mid-compaction never damages the source.

        When compacting in place over the file this facade was opened from,
        the database is reloaded afterwards so subsequent operations see the
        fresh file. When compacting to a different path, the facade keeps
        working on its current state.

        Args:
            output_path (Optional[str], optional): Where to write the
                compacted file. Defaults to None (compact in place using
                the file this facade was opened from).

        Returns:
            tuple[int, int]: Document counts (before, after).

        Raises:
            ValueError: If no target file is known (compact a project by
                dumping it first), or the source is missing or invalid.
        """

        default_target: Optional[str] = self.database.page_manager.storage_path
        target: str = output_path if output_path is not None else (default_target or "")

        if not target:
            raise ValueError(
                "No database file to compact: this facade was loaded from a "
                "project (pass output_path, or dump it first)"
            )

        in_place: bool = output_path is None or output_path == default_target

        before: int
        after: int
        before, after = compact_database(
            storage_path=target,
            output_path=output_path,
        )

        if in_place and default_target is not None:
            # Reload so cached pages of the pre-compaction file are dropped
            self.database = load_database(storage_path=default_target)
            self._globals = None

        return before, after

    # --- Code execution ---

    def execute_schema(self, schema_text: str) -> None:
        """
        Parses and resolves .nastruct schema source into the database.

        New structs and functions become available immediately; re-declaring
        an attribute that already exists raises a duplicate error.

        Args:
            schema_text (str): Raw .nastruct source text.
        """

        parser: NaStructParser = NaStructParser(database=self.database)

        parser.parse_file(content=schema_text)
        parser.resolve()

        # New structs and functions must appear in the runtime globals
        self._globals = None

    def execute_data(self, data_text: str) -> None:
        """
        Parses and executes .nadata source (instances and scripts).

        Instance declarations are created and bound, and script statements
        (function calls, queries, prints) execute immediately.

        Args:
            data_text (str): Raw .nadata source text.
        """

        loader: NaDataLoader = NaDataLoader(database=self.database)

        loader.load_file(content=data_text)

        # New instances and symbols must appear in the runtime globals
        self._globals = None

    def execute_script_file(
        self,
        file_path: str,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Runs one #manual script file with its parameter prelude.

        Args:
            file_path (str): Script path (resolved against the session
                folder when the app was opened as a folder session).
            params (Optional[dict[str, Any]], optional): Parameter values
                (strings or natives). Defaults to None.

        Raises:
            ValueError: If the file is missing, not #manual, or the
                parameters mismatch the declared signature.
        """

        resolved: Path = Path(file_path)

        if not resolved.is_file() and self.session_folder is not None:
            resolved = Path(self.session_folder) / file_path

        if not resolved.is_file():
            raise ValueError(f"Script file not found: '{file_path}'")

        folder: Path = (
            Path(self.session_folder)
            if self.session_folder is not None
            else resolved.parent
        )

        script: ScriptFile = parse_script_file(path=resolved, folder=folder)

        if script.kind != MANUAL:
            raise ValueError(
                f"'{script.rel_path}' is #{script.kind}, not #manual: "
                f"only #manual scripts run explicitly"
            )

        loader: NaDataLoader = NaDataLoader(database=self.database)

        loader.prelude = build_prelude(
            params=script.params,
            provided=params or {},
        )

        try:
            loader.load_file(content=script.text)

        finally:
            loader.prelude = {}

        # New instances and symbols must appear in the runtime globals
        self._globals = None

    def execute(self, text: str) -> None:
        """
        Executes Nadata source, auto detecting schema vs data content.

        Text containing top level 'class' or 'fn' declarations is treated as
        .nastruct schema source, everything else as .nadata data source.

        Args:
            text (str): Raw Nadata source text.
        """

        if re.search(r"(?:^|\n)\s*(?:class|fn)\s", text):
            self.execute_schema(schema_text=text)

        else:
            self.execute_data(data_text=text)

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
