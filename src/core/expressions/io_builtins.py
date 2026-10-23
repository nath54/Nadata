"""
Session I/O builtins: file and settings access from .nadata scripts.

All paths resolve against the project root (the session folder recorded
as the database's io_base); scripts without a project root resolve
against the current working directory. File effects are immediate —
unlike database writes, they are NOT gated by flush().

Contract summary:

- read_text / write_text: whole-file Str; writes overwrite.
- read_lines / append_text: Array[Str] / one appended line.
- read_json / write_json: objects of scalars, arrays of scalars, or
  scalars — homogeneous by construction (mixed JSON value types raise);
  writes overwrite.
- read_csv / write_csv: Array[dict[Str, Str]] — stringly-typed by
  design (no inference); writes overwrite.
- exists / file_hash / list_dir: Bool / sha256 hex Str / sorted
  Array[Str] of entry names.
- get_setting / set_setting: opt-in per-world persistence markers
  (scalars only); settings ride along in the next flush()'s meta doc.
"""

# Import Modules
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from src.core.expressions.scope import ScopeAwareFunction
from src.core.types.types import (
    Float64,
    Int64,
    TypeArray,
    TypeBool,
    TypeDict,
    TypeStr,
)
from src.core.values.values import (
    Value,
    ValueArray,
    ValueBool,
    ValueDict,
    ValueFloat,
    ValueInt,
    ValueStr,
)


def _resolve_path(base: Optional[str], raw: Any) -> Path:
    """
    Resolves a script path against the project root.

    When a project root is set (session folders), paths are confined
    to it: absolute paths and '..' traversal escaping the root are
    rejected — scripts can only touch files inside their own project.
    Without a project root, paths resolve against the current working
    directory (the operator's own machine, no confinement).

    Args:
        base (Optional[str]): Project root (None resolves against the
            current working directory).
        raw (Any): The argument as passed (must be a string).

    Returns:
        Path: Absolute or base-relative path.

    Raises:
        TypeError: If the path argument is not a string.
        PermissionError: If the path escapes the project root.
    """

    if not isinstance(raw, str):
        raise TypeError(
            f"Path must be a string, got '{type(raw).__name__}'"
        )

    path: Path = Path(raw)

    if base is None:
        return path

    root: Path = Path(base).resolve()

    # Absolute paths resolve directly; relative ones against the root
    resolved: Path = path.resolve() if path.is_absolute() else (
        (root / path).resolve()
    )

    # Sandbox: the resolved path must stay inside the project root
    if resolved != root and root not in resolved.parents:
        raise PermissionError(
            f"Path '{raw}' escapes the project root '{base}' — scripts "
            f"may only access files inside their own project"
        )

    return resolved


def _scalar_type_of(raw: Any) -> Any:
    """
    Maps a JSON scalar to its schema type.

    Args:
        raw (Any): JSON scalar (bool, int, float, str).

    Returns:
        Any: The corresponding schema Type.

    Raises:
        ValueError: If the value is not a JSON scalar.
    """

    if isinstance(raw, bool):
        return TypeBool()

    if isinstance(raw, int):
        return Int64

    if isinstance(raw, float):
        return Float64

    if isinstance(raw, str):
        return TypeStr()

    raise ValueError(
        f"JSON values must be scalars (bool/int/float/str), "
        f"got '{type(raw).__name__}'"
    )


def _wrap_scalar(raw: Any) -> Value:
    """
    Wraps a JSON scalar into its typed Value.

    Args:
        raw (Any): JSON scalar.

    Returns:
        Value: The typed scalar Value.
    """

    if isinstance(raw, bool):
        return ValueBool(value_type=TypeBool(), value=raw)

    if isinstance(raw, int):
        return ValueInt(value_type=Int64, value=raw)

    if isinstance(raw, float):
        return ValueFloat(value_type=Float64, value=raw)

    return ValueStr(value_type=TypeStr(), value=str(raw))


def _scalar_family(raw: Any) -> str:
    """
    Returns the homogeneity family of a JSON scalar.

    Int and float share the number family (a mixed int/float object
    is typed Float64), bool and str are their own families.

    Args:
        raw (Any): JSON scalar.

    Returns:
        str: 'bool', 'number', or 'str'.

    Raises:
        ValueError: On nested (non-scalar) values.
    """

    if not isinstance(raw, (bool, int, float, str)):
        raise ValueError(
            "read_json: nested structures are not supported — objects "
            "and arrays must directly contain scalars; stringify or "
            "split the file"
        )

    if isinstance(raw, bool):
        return "bool"

    if isinstance(raw, (int, float)):
        return "number"

    return "str"


def _json_to_value(data: Any) -> Value:
    """
    Converts a decoded JSON document into a typed Value.

    Supported shapes: scalars, arrays of scalars, and objects of
    scalars. Homogeneity is required — a mixed-type object or array
    raises with guidance (normalize the JSON, stringify, or split it).

    Args:
        data (Any): The json.load result.

    Returns:
        Value: The typed Value.

    Raises:
        ValueError: On nested or mixed-type structures.
    """

    if isinstance(data, dict):

        if not data:
            return ValueDict(
                value_type=TypeDict(value_type=TypeStr()),
                entries={},
            )

        families: set[str] = {
            _scalar_family(raw=value) for value in data.values()
        }

        if len(families) > 1:
            raise ValueError(
                "read_json: object values must all be the same scalar "
                f"family, got {sorted(families)} — normalize the JSON, "
                f"stringify, or split it across files"
            )

        family: str = families.pop()

        if family == "number" and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in data.values()
        ):
            value_type: Any = Int64

        elif family == "number":
            value_type = Float64

        else:
            value_type = _scalar_type_of(next(iter(data.values())))

        return ValueDict(
            value_type=TypeDict(value_type=value_type),
            entries=dict(data),
        )

    if isinstance(data, list):

        if not data:
            return ValueArray(
                value_type=TypeArray(value_type=TypeStr()),
                elements=[],
            )

        element_families: set[str] = {
            _scalar_family(raw=element) for element in data
        }

        if len(element_families) > 1:
            raise ValueError(
                "read_json: array elements must all be the same scalar "
                f"family, got {sorted(element_families)} — normalize the "
                f"JSON, stringify, or split it across files"
            )

        element_family: str = element_families.pop()

        if element_family == "number" and all(
            isinstance(element, int) and not isinstance(element, bool)
            for element in data
        ):
            element_type: Any = Int64

        elif element_family == "number":
            element_type = Float64

        else:
            element_type = _scalar_type_of(data[0])

        return ValueArray(
            value_type=TypeArray(value_type=element_type),
            elements=list(data),
        )

    return _wrap_scalar(raw=data)


class IoFunction(ScopeAwareFunction):
    """
    Base for the I/O builtins: carries the database (for the project
    root and settings) and shared argument plumbing.
    """

    def __init__(self, database: Any) -> None:
        """
        Initializes the IoFunction instance.

        Args:
            database (Any): Database providing io_base and settings.
        """

        self.database: Any = database

    def _path(self, args: list[Value], index: int = 0) -> Path:
        """
        Resolves the path argument at the given position.

        Args:
            args (list[Value]: The call arguments.
            index (int, optional): Argument position. Defaults to 0.

        Returns:
            Path: The resolved path.
        """

        return _resolve_path(
            base=self.database.io_base,
            raw=args[index].get(),
        )


class ReadTextFunction(IoFunction):
    """
    Native read_text(path): the whole file as a Str.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueStr:
        """
        Reads the file.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueStr: The file content.

        Raises:
            TypeError: On wrong arity.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("read_text takes exactly one argument")

        return ValueStr(
            value_type=TypeStr(),
            value=self._path(args=args).read_text(encoding="utf-8"),
        )


class WriteTextFunction(IoFunction):
    """
    Native write_text(path, content): overwrites the file with the
    string. Returns the bytes written.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueInt:
        """
        Writes the file.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path and the content.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueInt: Bytes written.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("write_text takes exactly two arguments")

        path: Path = self._path(args=args)
        path.parent.mkdir(parents=True, exist_ok=True)

        content: str = str(args[1].get())

        written: int = path.write_text(content, encoding="utf-8")

        return ValueInt(value_type=Int64, value=written)


class ReadLinesFunction(IoFunction):
    """
    Native read_lines(path): the file's lines as Array[Str] (without
    line endings; a trailing newline does not add an empty entry).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueArray:
        """
        Reads the file's lines.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueArray: The lines.

        Raises:
            TypeError: On wrong arity.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("read_lines takes exactly one argument")

        content: str = self._path(args=args).read_text(encoding="utf-8")

        lines: list[str] = content.splitlines()

        return ValueArray(
            value_type=TypeArray(value_type=TypeStr()),
            elements=lines,
        )


class AppendTextFunction(IoFunction):
    """
    Native append_text(path, line): appends one line (a newline is
    added when missing). Creates the file when absent. Returns the
    bytes written.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueInt:
        """
        Appends the line.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path and the line.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueInt: Bytes written.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("append_text takes exactly two arguments")

        path: Path = self._path(args=args)
        path.parent.mkdir(parents=True, exist_ok=True)

        line: str = str(args[1].get())

        if not line.endswith("\n"):
            line += "\n"

        with open(path, "a", encoding="utf-8") as file:
            written: int = file.write(line)

        return ValueInt(value_type=Int64, value=written)


class ReadJsonFunction(IoFunction):
    """
    Native read_json(path): objects of scalars, arrays of scalars, or
    scalars. Mixed-type structures raise (see _json_to_value).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Value:
        """
        Reads and converts the JSON document.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Value: The typed Value (ValueDict / ValueArray / scalar).

        Raises:
            TypeError: On wrong arity.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("read_json takes exactly one argument")

        data: Any = json.loads(
            self._path(args=args).read_text(encoding="utf-8")
        )

        return _json_to_value(data=data)


class WriteJsonFunction(IoFunction):
    """
    Native write_json(path, value): serializes the value (dict,
    array, or scalar) as JSON, overwriting the file. Returns the
    bytes written.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueInt:
        """
        Writes the JSON document.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path and the value.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueInt: Bytes written.

        Raises:
            TypeError: On wrong arity.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("write_json takes exactly two arguments")

        path: Path = self._path(args=args)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: str = json.dumps(
            obj=_jsonable(raw=args[1].get()),
            indent=2,
        ) + "\n"

        written: int = path.write_text(payload, encoding="utf-8")

        return ValueInt(value_type=Int64, value=written)


def _jsonable(raw: Any) -> Any:
    """
    Converts a raw script value to its JSON-serializable form.

    Args:
        raw (Any): Raw payload from a Value (dict entries / array
            elements / scalar are already JSON-native).

    Returns:
        Any: The JSON-serializable payload.

    Raises:
        TypeError: On non-serializable payloads.
    """

    if isinstance(raw, dict):
        return {
            str(key): _jsonable(raw=value)
            for key, value in raw.items()
        }

    if isinstance(raw, (list, tuple)):
        return [_jsonable(raw=element) for element in raw]

    if isinstance(raw, (bool, int, float, str)) or raw is None:
        return raw

    raise TypeError(
        f"write_json: values must be dicts, arrays, or scalars, "
        f"got '{type(raw).__name__}'"
    )


class ReadCsvFunction(IoFunction):
    """
    Native read_csv(path): the CSV as Array[dict[Str, Str]]. The
    first row is the header; all values stay strings (no inference).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueArray:
        """
        Parses the CSV file.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueArray: Array of string-keyed, string-valued dicts.

        Raises:
            TypeError: On wrong arity.
            ValueError: On a ragged row (key mismatch with the header).
        """

        if len(args) != 1 or kwargs:
            raise TypeError("read_csv takes exactly one argument")

        lines: list[str] = self._path(args=args).read_text(
            encoding="utf-8"
        ).splitlines()

        if not lines:
            return ValueArray(
                value_type=TypeArray(value_type=TypeDict(
                    value_type=TypeStr(),
                    key_type=TypeStr(),
                )),
                elements=[],
            )

        header: list[str] = lines[0].split(",")

        rows: list[dict[str, str]] = []

        for line in lines[1:]:

            if not line:
                continue

            cells: list[str] = line.split(",")

            if len(cells) != len(header):
                raise ValueError(
                    f"read_csv: row has {len(cells)} cells but the "
                    f"header has {len(header)}"
                )

            rows.append(dict(zip(header, cells)))

        return ValueArray(
            value_type=TypeArray(value_type=TypeDict(
                value_type=TypeStr(),
                key_type=TypeStr(),
            )),
            elements=rows,
        )


class WriteCsvFunction(IoFunction):
    """
    Native write_csv(path, rows): serializes an Array[dict[Str, Str]]
    with the first row's keys as the header (all rows must share the
    key set). Returns the data rows written.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueInt:
        """
        Writes the CSV file.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path and the rows.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueInt: Data rows written (0 for an empty array).

        Raises:
            TypeError: On wrong arity or kinds.
            ValueError: On a row whose keys differ from the header.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("write_csv takes exactly two arguments")

        path: Path = self._path(args=args)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows: Any = args[1].get()

        if not isinstance(rows, list):
            raise TypeError(
                "write_csv expects an array of dicts, got "
                f"'{type(rows).__name__}'"
            )

        if not rows:
            path.write_text("", encoding="utf-8")

            return ValueInt(value_type=Int64, value=0)

        header: list[str] = [str(key) for key in rows[0].keys()]

        lines: list[str] = [",".join(header)]

        for row in rows:

            if not isinstance(row, dict):
                raise TypeError(
                    "write_csv rows must be dicts, got "
                    f"'{type(row).__name__}'"
                )

            if set(str(key) for key in row.keys()) != set(header):
                raise ValueError(
                    "write_csv: all rows must share the header's keys"
                )

            lines.append(",".join(
                str(row[key]) for key in rows[0].keys()
            ))

        written_rows: int = len(rows)

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return ValueInt(value_type=Int64, value=written_rows)


class ExistsFunction(IoFunction):
    """
    Native exists(path): whether the path exists (file or directory).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueBool:
        """
        Checks the path.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueBool: Existence.

        Raises:
            TypeError: On wrong arity.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("exists takes exactly one argument")

        return ValueBool(
            value_type=TypeBool(),
            value=self._path(args=args).exists(),
        )


class FileHashFunction(IoFunction):
    """
    Native file_hash(path): the file's sha256 as a hex Str. Use with
    get_setting / set_setting for opt-in input freshness.
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueStr:
        """
        Hashes the file.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The path.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueStr: The sha256 hex digest.

        Raises:
            TypeError: On wrong arity.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("file_hash takes exactly one argument")

        data: bytes = self._path(args=args).read_bytes()

        return ValueStr(
            value_type=TypeStr(),
            value=hashlib.sha256(data).hexdigest(),
        )


class ListDirFunction(IoFunction):
    """
    Native list_dir(path): sorted entry names as Array[Str].
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> ValueArray:
        """
        Lists the directory.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The directory path.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            ValueArray: Sorted entry names.

        Raises:
            TypeError: On wrong arity.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("list_dir takes exactly one argument")

        names: list[str] = sorted(
            entry.name for entry in self._path(args=args).iterdir()
        )

        return ValueArray(
            value_type=TypeArray(value_type=TypeStr()),
            elements=names,
        )


class GetSettingFunction(IoFunction):
    """
    Native get_setting(name): the per-world setting value, or None
    when unset. Scalars only (persisted in the world's meta doc).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> Any:
        """
        Reads the setting.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The setting name.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Returns:
            Any: The typed scalar Value, or None when unset.

        Raises:
            TypeError: On wrong arity or kinds.
        """

        if len(args) != 1 or kwargs:
            raise TypeError("get_setting takes exactly one argument")

        name: Any = args[0].get()

        if not isinstance(name, str):
            raise TypeError("Setting name must be a string")

        if name not in self.database.settings:
            return None

        return _wrap_scalar(raw=self.database.settings[name])


class SetSettingFunction(IoFunction):
    """
    Native set_setting(name, value): records a per-world scalar
    setting. Persisted by the next flush() (settings ride along in
    the meta document). Intended for freshness markers:
    set_setting("data_hash", file_hash("data.csv")).
    """

    def invoke(
        self,
        scope: Any,
        args: list[Value],
        kwargs: dict[str, Value],
    ) -> None:
        """
        Writes the setting.

        Args:
            scope (Any): Live execution scope (unused).
            args (list[Value]): The setting name and scalar value.
            kwargs (dict[str, Value]): Keyword arguments (rejected).

        Raises:
            TypeError: On wrong arity, a non-string name, or a
                non-scalar value.
        """

        if len(args) != 2 or kwargs:
            raise TypeError("set_setting takes exactly two arguments")

        name: Any = args[0].get()
        value: Any = args[1].get()

        if not isinstance(name, str):
            raise TypeError("Setting name must be a string")

        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                "Setting values must be scalars (bool/int/float/str), "
                f"got '{type(value).__name__}'"
            )

        self.database.settings[name] = value

        return None


def register_io_builtins(
    globals_dict: dict[str, Any],
    database: Any,
) -> None:
    """
    Registers the I/O builtins into the runtime globals.

    Args:
        globals_dict (dict[str, Any]): Runtime globals to extend.
        database (Any): Database providing io_base and settings.
    """

    globals_dict["read_text"] = ReadTextFunction(database=database)
    globals_dict["write_text"] = WriteTextFunction(database=database)
    globals_dict["read_lines"] = ReadLinesFunction(database=database)
    globals_dict["append_text"] = AppendTextFunction(database=database)
    globals_dict["read_json"] = ReadJsonFunction(database=database)
    globals_dict["write_json"] = WriteJsonFunction(database=database)
    globals_dict["read_csv"] = ReadCsvFunction(database=database)
    globals_dict["write_csv"] = WriteCsvFunction(database=database)
    globals_dict["exists"] = ExistsFunction(database=database)
    globals_dict["file_hash"] = FileHashFunction(database=database)
    globals_dict["list_dir"] = ListDirFunction(database=database)
    globals_dict["get_setting"] = GetSettingFunction(database=database)
    globals_dict["set_setting"] = SetSettingFunction(database=database)
