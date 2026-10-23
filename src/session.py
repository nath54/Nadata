"""
Folder sessions: directive-classified .nadata files (#init/#auto/#manual)
loaded against a persistent world database.

A project folder is the SYSTEM (schemas re-merged every open, always
idempotent); the .ndb file is one WORLD of its data. #init files run
once per world (content-hash markers in settings), #auto files run on
every open (nothing persists unless the script flushes), and #manual
files run only when explicitly invoked — with a typed parameter
signature declared in the directive itself.
"""

# Import Modules
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Any, Optional

from src.parser.nadata_loader.loader import NaDataLoader

from src.core.database.database import NaDataDatabase
from src.core.database.persistence.dumper import (
    dump_database,
    flush_dirty_instances,
)
from src.core.database.persistence.loader import load_database
from src.core.types.types import (
    Float32,
    Float64,
    Int32,
    Int64,
    Type,
    TypeBool,
    TypeFloat,
    TypeInt,
    TypeStr,
)
from src.core.values.values import (
    Value,
    ValueBool,
    ValueFloat,
    ValueInt,
    ValueStr,
)

# Default world database file name inside the project folder
DEFAULT_DB_FILENAME: str = "nadata.ndb"

# Directive first-line pattern: '#init', '#auto', '#manual(...)'
_DIRECTIVE_RE: re.Pattern[str] = re.compile(
    r"^#\s*(init|auto|manual)\s*(?:\((.*)\))?\s*$"
)

# One manual parameter: 'name: Type' or 'name: Type = default'
_PARAM_RE: re.Pattern[str] = re.compile(
    r"^(\w+)\s*:\s*(\w+)\s*(?:=\s*(.+))?$"
)

# Directive kinds
INIT: str = "init"
AUTO: str = "auto"
MANUAL: str = "manual"


class ManualParam:
    """
    One declared parameter of a #manual script.

    Attributes:
        name (str): Parameter name.
        type_name (str): Declared scalar type ('Int', 'Float', 'Str',
            'Bool').
        default_text (Optional[str]): Literal default text, or None for
            a required parameter.
    """

    __slots__ = ("name", "type_name", "default_text")

    def __init__(self, name: str, type_name: str, default_text: Optional[str]) -> None:
        """
        Initializes the ManualParam instance.

        Args:
            name (str): Parameter name.
            type_name (str): Declared type.
            default_text (Optional[str]): Literal default, or None.
        """

        self.name: str = name
        self.type_name: str = type_name
        self.default_text: Optional[str] = default_text


class ScriptFile:
    """
    One classified .nadata file.

    Attributes:
        rel_path (str): Folder-relative posix path.
        absolute (Path): Absolute file path.
        kind (str): Directive kind ('init', 'auto', or 'manual').
        params (list[ManualParam]): Declared manual parameters.
        text (str): Full raw text (the directive line is a comment).
        content_hash (str): SHA-256 of the raw text.
    """

    __slots__ = (
        "rel_path",
        "absolute",
        "kind",
        "params",
        "text",
        "content_hash",
    )

    def __init__(
        self,
        rel_path: str,
        absolute: Path,
        kind: str,
        params: list[ManualParam],
        text: str,
        content_hash: str,
    ) -> None:
        """
        Initializes the ScriptFile instance.

        Args:
            rel_path (str): Folder-relative posix path.
            absolute (Path): Absolute path.
            kind (str): Directive kind.
            params (list[ManualParam]): Declared parameters.
            text (str): Raw text.
            content_hash (str): SHA-256 of the text.
        """

        self.rel_path: str = rel_path
        self.absolute: Path = absolute
        self.kind: str = kind
        self.params: list[ManualParam] = params
        self.text: str = text
        self.content_hash: str = content_hash

    def marker_key(self) -> str:
        """
        Returns the init marker key (path plus content hash).

        Returns:
            str: The settings key.
        """

        return f"{self.rel_path}:{self.content_hash}"


class FolderResult:
    """
    Outcome of one folder session pipeline run.

    Attributes:
        folder (str): Project folder.
        db_path (str): World database file.
        created_fresh (bool): True when the world was created now.
        init_ran (list[str]): Init scripts that executed.
        init_skipped (list[str]): Init scripts skipped (marker hit).
        init_errors (list[tuple[str, str]]): Init failures (path, error).
        auto_ran (list[str]): Auto scripts that executed.
        auto_errors (list[tuple[str, str]]): Auto failures.
    """

    def __init__(
        self,
        folder: str,
        db_path: str,
        created_fresh: bool,
    ) -> None:
        """
        Initializes the FolderResult instance.

        Args:
            folder (str): Project folder.
            db_path (str): World database file.
            created_fresh (bool): Whether the world is new.
        """

        self.folder: str = folder
        self.db_path: str = db_path
        self.created_fresh: bool = created_fresh
        self.init_ran: list[str] = []
        self.init_skipped: list[str] = []
        self.init_errors: list[tuple[str, str]] = []
        self.auto_ran: list[str] = []
        self.auto_errors: list[tuple[str, str]] = []
        self.instance_count: int = 0

    def ok(self) -> bool:
        """
        Returns whether the pipeline finished without errors.

        Returns:
            bool: True when no init or auto errors occurred.
        """

        return not self.init_errors and not self.auto_errors

    def summary(self) -> str:
        """
        Renders a one-line pipeline report.

        Returns:
            str: The summary line.
        """

        parts: list[str] = [
            f"world={'created' if self.created_fresh else 'opened'} "
            f"({self.db_path})",
            f"init {len(self.init_ran)} ran / {len(self.init_skipped)} skipped"
            f" / {len(self.init_errors)} failed",
            f"auto {len(self.auto_ran)} ran / {len(self.auto_errors)} failed",
        ]

        return "; ".join(parts)


def parse_script_file(path: Path, folder: Path) -> ScriptFile:
    """
    Reads a .nadata file and classifies it by its directive first line.

    The first non-blank line matching '#init', '#auto', or
    '#manual(...)' sets the kind; unmarked files are #manual with no
    parameters (automatic execution is opt-in).

    Args:
        path (Path): Absolute file path.
        folder (Path): Project folder (for the relative path).

    Returns:
        ScriptFile: The classified file.
    """

    text: str = path.read_text(encoding="utf-8")

    kind: str = MANUAL
    params: list[ManualParam] = []

    for line in text.splitlines():

        stripped: str = line.strip()

        if not stripped:
            continue

        match: Optional[re.Match[str]] = _DIRECTIVE_RE.match(stripped)

        if match is not None:
            kind = match.group(1)

            if match.group(2):
                params = _parse_params(signature=match.group(2))

        break

    return ScriptFile(
        rel_path=path.relative_to(folder).as_posix(),
        absolute=path,
        kind=kind,
        params=params,
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _parse_params(signature: str) -> list[ManualParam]:
    """
    Parses a #manual parameter signature.

    Args:
        signature (str): Comma-separated 'name: Type = default' items.

    Returns:
        list[ManualParam]: Declared parameters in order.

    Raises:
        ValueError: On a malformed parameter declaration.
    """

    params: list[ManualParam] = []

    for item in signature.split(","):

        item = item.strip()

        if not item:
            continue

        match: Optional[re.Match[str]] = _PARAM_RE.match(item)

        if match is None:
            raise ValueError(
                f"Malformed manual parameter '{item}' "
                f"(expected 'name: Type = default')"
            )

        params.append(
            ManualParam(
                name=match.group(1),
                type_name=match.group(2),
                default_text=match.group(3),
            )
        )

    return params


def scan_folder(folder: Path) -> tuple[list[Path], list[ScriptFile]]:
    """
    Scans a folder for schema files and classified scripts.

    Args:
        folder (Path): Project folder.

    Returns:
        tuple[list[Path], list[ScriptFile]]: Sorted .nastruct paths and
            sorted classified .nadata files.
    """

    schema_files: list[Path] = sorted(folder.rglob("*.nastruct"))
    scripts: list[ScriptFile] = [
        parse_script_file(path=path, folder=folder)
        for path in sorted(folder.rglob("*.nadata"))
    ]

    return schema_files, scripts


def folder_is_session(folder: Path) -> bool:
    """
    Checks whether a folder uses session semantics.

    A folder is a session when any .nadata file carries a directive or
    the default world database exists.

    Args:
        folder (Path): Project folder.

    Returns:
        bool: True for session semantics.
    """

    if (folder / DEFAULT_DB_FILENAME).exists():
        return True

    for path in sorted(folder.rglob("*.nadata")):

        for line in path.read_text(encoding="utf-8").splitlines():

            stripped: str = line.strip()

            if not stripped:
                continue

            if _DIRECTIVE_RE.match(stripped) is not None:
                return True

            break

    return False


# Accepted parameter type names -> runtime schema types
_PARAM_TYPES: dict[str, Type] = {
    "Int": Int64,
    "Int32": Int32,
    "Int64": Int64,
    "Float": Float64,
    "Float32": Float32,
    "Float64": Float64,
    "Str": TypeStr(),
    "Bool": TypeBool(),
}


def coerce_param(type_name: str, raw: Any) -> Any:
    """
    Coerces one parameter value to its declared scalar type.

    Args:
        type_name (str): 'Int', 'Float', 'Str', or 'Bool'.
        raw (Any): The value (CLI strings or native Python values).

    Returns:
        Any: The coerced Python value.

    Raises:
        ValueError: On an unknown type or an uncoercible value.
    """

    if type_name not in _PARAM_TYPES:
        raise ValueError(
            f"Unsupported parameter type '{type_name}' "
            f"(supported: {', '.join(_PARAM_TYPES)})"
        )

    if type_name.startswith("Int"):

        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            raise ValueError(f"Cannot coerce '{raw!r}' to {type_name}")

        try:
            return int(raw)

        except ValueError:
            raise ValueError(f"Cannot coerce '{raw!r}' to {type_name}") from None

    if type_name.startswith("Float"):

        if not isinstance(raw, (int, float, str)):
            raise ValueError(f"Cannot coerce '{raw!r}' to {type_name}")

        try:
            return float(raw)

        except ValueError:
            raise ValueError(f"Cannot coerce '{raw!r}' to {type_name}") from None

    if type_name == "Str":
        return raw if isinstance(raw, str) else str(raw)

    if type_name == "Bool":

        if isinstance(raw, bool):
            return raw

        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            return raw.lower() == "true"

        raise ValueError(f"Cannot coerce '{raw!r}' to Bool")

    raise ValueError(
        f"Unsupported parameter type '{type_name}' "
        f"(supported: {', '.join(_PARAM_TYPES)})"
    )


def coerce_param_literal(type_name: str, text: str) -> Any:
    """
    Coerces a directive default literal into its Python value.

    Args:
        type_name (str): Declared type.
        text (str): Literal text (quotes stripped for Str).

    Returns:
        Any: The Python value.
    """

    if type_name == "Str":
        return text.strip().strip('"').strip("'")

    if type_name == "Bool":
        return text.strip().lower() == "true"

    return coerce_param(type_name=type_name, raw=text.strip())


def wrap_scalar(type_name: str, value: Any) -> Value:
    """
    Wraps a coerced scalar into the matching runtime Value.

    Args:
        type_name (str): Declared type.
        value (Any): Coerced Python value.

    Returns:
        Value: The runtime value.

    Raises:
        ValueError: On an unknown type.
    """

    schema_type: Type = _PARAM_TYPES.get(type_name, TypeStr())

    if isinstance(schema_type, TypeInt):
        return ValueInt(value_type=schema_type, value=int(value))

    if isinstance(schema_type, TypeFloat):
        return ValueFloat(value_type=schema_type, value=float(value))

    if isinstance(schema_type, TypeBool):
        return ValueBool(value_type=schema_type, value=bool(value))

    return ValueStr(value_type=TypeStr(), value=str(value))


def build_prelude(
    params: list[ManualParam],
    provided: dict[str, Any],
) -> dict[str, Value]:
    """
    Builds the pre-bound parameter scope for a #manual script.

    Provided names must be declared; missing names fall back to
    defaults; required-but-missing raises with the full signature.

    Args:
        params (list[ManualParam]): Declared parameters.
        provided (dict[str, Any]): Invocation values (strings or
            natives).

    Returns:
        dict[str, Value]: Pre-bound scope values.

    Raises:
        ValueError: On unknown names, bad values, or missing required
            parameters.
    """

    declared: dict[str, ManualParam] = {
        param.name: param for param in params
    }

    for name in provided:
        if name not in declared:
            raise ValueError(
                f"Unknown parameter '{name}' "
                f"(expected: {', '.join(declared) or 'none'})"
            )

    prelude: dict[str, Value] = {}

    for param in params:

        if param.name in provided:
            raw: Any = provided[param.name]

        elif param.default_text is not None:
            raw = coerce_param_literal(
                type_name=param.type_name,
                text=param.default_text,
            )

        else:
            raise ValueError(
                f"Missing required parameter '{param.name}' "
                f"({'#manual(' + ', '.join(p.name + ': ' + p.type_name for p in params) + ')'})"
            )

        prelude[param.name] = wrap_scalar(
            type_name=param.type_name,
            value=coerce_param(type_name=param.type_name, raw=raw),
        )

    return prelude


def _execute_schema_into(database: NaDataDatabase, path: Path) -> None:
    """
    Parses and merges one schema file into the database.

    Args:
        database (NaDataDatabase): Target database.
        path (Path): Schema file.
    """

    from src.parser.nastruct_parser.nastruct_parser import NaStructParser

    parser: NaStructParser = NaStructParser(database=database)

    # Session schemas re-merge on every open: re-declared members replace,
    # new ones are added, existing tables and rows stay untouched
    parser.merge = True

    parser.parse_file(content=path.read_text(encoding="utf-8"))
    parser.resolve()


def _instance_count(database: NaDataDatabase) -> int:
    """
    Counts decoded plus pending instances across all tables.

    Args:
        database (NaDataDatabase): The world database.

    Returns:
        int: The instance count.
    """

    total: int = 0

    for table in database.tables_by_type_id.values():

        decoder: Any = getattr(table.instances, "decoder", None)

        if decoder is None:
            total += len(dict.keys(table.instances))

            continue

        # Row-backed entities include already-decoded ones, so the
        # distinct count is the union of decoded and pending ids
        decoded_entities: set[int] = {
            instance.entity_id.value
            for instance in table.instances.values()
        }

        total += len(decoded_entities | decoder.pending_entity_ids())

    return total


def load_folder_session(
    folder_path: str,
    db_path: Optional[str] = None,
    run_auto: bool = True,
    force_init: bool = False,
) -> tuple[NaDataDatabase, FolderResult]:
    """
    Runs the folder session pipeline and returns the world database.

    Schemas are re-merged; #init scripts run once per world (content
    hash markers, flushed on success); #auto scripts run when asked;
    #manual scripts are skipped. A fresh world is only persisted when
    every init script succeeded (failed seeding leaves nothing behind).

    Args:
        folder_path (str): Project folder.
        db_path (Optional[str], optional): World database file
            (relative paths resolve against the folder). Defaults to
            '<folder>/nadata.ndb'.
        run_auto (bool, optional): Whether to run #auto scripts.
            Defaults to True.
        force_init (bool, optional): Re-run init scripts whose markers
            exist. Defaults to False.

    Returns:
        tuple[NaDataDatabase, FolderResult]: The world and the report.

    Raises:
        ValueError: If the folder does not exist.
    """

    folder: Path = Path(folder_path)

    if not folder.is_dir():
        raise ValueError(f"Not a folder: '{folder_path}'")

    world_path: Path = (
        Path(db_path)
        if db_path is not None and Path(db_path).is_absolute()
        else folder / (db_path or DEFAULT_DB_FILENAME)
    )

    schema_files, scripts = scan_folder(folder=folder)

    init_files: list[ScriptFile] = [
        script for script in scripts if script.kind == INIT
    ]
    auto_files: list[ScriptFile] = [
        script for script in scripts if script.kind == AUTO
    ]

    created_fresh: bool = not world_path.exists()

    result: FolderResult = FolderResult(
        folder=str(folder),
        db_path=str(world_path),
        created_fresh=created_fresh,
    )

    database: NaDataDatabase = (
        NaDataDatabase()
        if created_fresh
        else load_database(storage_path=str(world_path))
    )

    # Script paths (read_text, write_json, ...) resolve against the
    # project root
    database.io_base = str(folder)

    for schema_file in schema_files:
        _execute_schema_into(database=database, path=schema_file)

    init_failed: bool = False

    for script in init_files:

        marker: str = script.marker_key()
        markers: dict[str, Any] = database.settings.setdefault(
            "init_done", {}
        )

        if marker in markers and not force_init:
            result.init_skipped.append(script.rel_path)

            continue

        try:
            loader: NaDataLoader = NaDataLoader(database=database)
            loader.load_file(content=script.text)

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            NameError,
            SyntaxError,
        ) as error:
            init_failed = True
            result.init_errors.append(
                (script.rel_path, f"{type(error).__name__}: {error}")
            )

            continue

        markers[marker] = True
        result.init_ran.append(script.rel_path)

        if not created_fresh:
            world_path.parent.mkdir(parents=True, exist_ok=True)

            flush_dirty_instances(
                database=database, storage_path=str(world_path)
            )

    # A failed #init aborts the rest of the open: pass-1 shells from the
    # broken script may linger in memory, so an existing world is
    # reloaded fresh from disk (index-only, cheap) and #auto is skipped —
    # nothing from the broken open can reach the file
    if init_failed and not created_fresh:
        database = load_database(storage_path=str(world_path))

    if run_auto and not init_failed:

        for script in auto_files:

            try:
                loader = NaDataLoader(database=database)
                loader.load_file(content=script.text)

            except (
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                NameError,
                SyntaxError,
            ) as error:
                result.auto_errors.append(
                    (script.rel_path, f"{type(error).__name__}: {error}")
                )

            else:
                result.auto_ran.append(script.rel_path)

    # A fresh world persists only when its seeding succeeded; failed
    # init leaves nothing behind (the memory-model transaction)
    if created_fresh and not init_failed:
        world_path.parent.mkdir(parents=True, exist_ok=True)

        dump_database(database=database, storage_path=str(world_path))

        # The dump writes the file; the database must know its storage
        # so later flush() calls (language builtin) target it
        database.storage_path = str(world_path)

    result.instance_count = _instance_count(database=database)

    return database, result


def execute_manual_script(
    database: NaDataDatabase,
    script: ScriptFile,
    folder: Path,
    provided: Optional[dict[str, Any]] = None,
) -> None:
    """
    Runs one #manual script with its parameter prelude.

    Args:
        database (NaDataDatabase): World database.
        script (ScriptFile): The classified script.
        folder (Path): Project folder (unused; kept for signature
            symmetry with path resolution callers).
        provided (Optional[dict[str, Any]], optional): Parameter values.
        Defaults to None.

    Raises:
        ValueError: If the script is not #manual or parameters mismatch.
    """

    if script.kind != MANUAL:
        raise ValueError(
            f"'{script.rel_path}' is #{script.kind}, not #manual"
        )

    loader: NaDataLoader = NaDataLoader(database=database)

    loader.prelude = build_prelude(
        params=script.params,
        provided=provided or {},
    )

    try:
        loader.load_file(content=script.text)

    finally:
        loader.prelude = {}
