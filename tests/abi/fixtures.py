"""
Reusable engine scenarios for the ABI conformance suite.

Every builder returns a freshly-loaded facade so tests never share
mutable state. The builders also define the exact engine paths the
future C shim uses (e.g. `where_str` documents the parse + where_expr
sequence that `nd_query_where_str` will perform).
"""

# Import Modules
import shutil
from pathlib import Path
from typing import Any

from nadata.api import Nadata
from nadata.core.expressions.scope import ExecutionScope
from nadata.parser.expressions.expression_parser import ExpressionParser

# Relative to the repository root (pytest rootdir).
EX0_PATH: str = "./examples/ex0_simple"

# Expected counts for ex0_simple, asserted in the fixture test and
# relied on by many conformance cases.
EX0_HUMANS: int = 3          # tom, jean, louise
EX0_ANIMALS: int = 1         # white_cat (Cat is an Animal subclass)
EX0_ENTITIES: int = 4        # polymorphic: humans + animals
EX0_TOTAL: int = 6           # + knife's ObjectType row + the ObjectInstance


def loaded_ex0() -> Nadata:
    """
    Loads a fresh ex0_simple facade.

    Returns:
        Nadata: Newly loaded facade (never shared between tests).
    """

    return Nadata.load(path=EX0_PATH)


def dumped_ex0(directory: Path, name: str = "store.ndb") -> str:
    """
    Dumps a fresh ex0_simple database into a new .ndb file.

    Args:
        directory (Path): Target directory (created if missing).
        name (str): Database file name.

    Returns:
        str: The storage path handed to dump().
    """

    directory.mkdir(parents=True, exist_ok=True)

    storage_path: str = str(directory / name)
    dumped: Nadata = loaded_ex0()
    dumped.dump(storage_path=storage_path)

    return storage_path


def numeric_project(directory: Path) -> Nadata:
    """
    Builds a tiny project with numeric attributes for aggregations.

    Layout matches the engine's project expectations: `structs/` first,
    then `data/`. Three boxes: weights 1.5 / 2.5 / 3.5, counts 2 / 3 / 5.

    Args:
        directory (Path): Project folder (created if missing).

    Returns:
        Nadata: The loaded facade over the project.
    """

    schema: str = (
        "class Box {\n"
        '    label: Str = "";\n'
        "    weight: Float64 = 0.0;\n"
        "    count: Int32 = 0;\n"
        "}\n"
    )
    data: str = (
        'a = Box(label = "a", weight = 1.5, count = 2);\n'
        'b = Box(label = "b", weight = 2.5, count = 3);\n'
        'c = Box(label = "c", weight = 3.5, count = 5);\n'
    )

    _write(directory, "structs/boxes.nastruct", schema)
    _write(directory, "data/boxes.nadata", data)

    return Nadata.load(path=str(directory))


def session_project(directory: Path) -> tuple[Nadata, Path]:
    """
    Builds a minimal folder session with an #init seed and a #manual
    script that captures its parameters into a Child row.

    Args:
        directory (Path): Session folder (created if missing).

    Returns:
        tuple[Nadata, Path]: The opened session facade and folder.
    """

    schema: str = (
        "class Child {\n"
        "    name: Str;\n"
        "    age: Int32 = 0;\n"
        "}\n"
    )
    seed: str = (
        "#init\n"
        'tom = Child(nid = "tom", name = "Tom", age = 6);\n'
        "flush();\n"
    )
    capture: str = (
        "#manual(v: Int = 7, name: Str = \"x\")\n"
        'captured = Child(nid = "captured", age = v, name = name);\n'
        "flush();\n"
    )

    _write(directory, "structs/schema.nastruct", schema)
    _write(directory, "data/seed.nadata", seed)
    _write(directory, "capture.nadata", capture)

    app: Nadata = Nadata.open_folder(folder_path=str(directory))

    return app, directory


def copied_ex0(directory: Path) -> Path:
    """
    Copies ex0_simple into a temp folder (for open_folder tests that
    must not write into the repository's examples/).

    Args:
        directory (Path): Destination parent (created if missing).

    Returns:
        Path: The copied project folder.
    """

    directory.mkdir(parents=True, exist_ok=True)
    destination: Path = directory / "proj"

    shutil.copytree(EX0_PATH, destination)

    return destination


def where_str(query_set: Any, var: str, text: str) -> Any:
    """
    Filters a QuerySet with a predicate written in the nadata language.

    This is the exact engine path `nd_query_where_str` will use: parse
    the text with the shared expression parser, then evaluate it per
    candidate through `where_expr`. Kept here so the conformance suite
    and the shim cannot drift.

    String literals use double quotes — single quotes are not string
    delimiters in the language (SyntaxError).

    Args:
        query_set (Any): The QuerySet to filter.
        var (str): Candidate variable name (e.g. "h").
        text (str): Predicate expression (e.g. 'h.name == "Tom"').

    Returns:
        Any: New filtered QuerySet.
    """

    expression: Any = ExpressionParser.parse_text(text)

    return query_set.where_expr(candidate_var=var, expr=expression,
                                scope=ExecutionScope())


def _write(folder: Path, rel: str, text: str) -> None:
    """
    Writes a UTF-8 file into a project folder (creating parents).

    Args:
        folder (Path): Project folder root.
        rel (str): Relative posix path.
        text (str): File content.
    """

    target: Path = folder / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
