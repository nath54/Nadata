"""
Schema file imports: `import "relative/path.nastruct";` declarations.

A .nastruct file may import other schema files at the top of its
source. Imported files are parsed before the importer (dependency
order), deduplicated across the whole parse (a library imported by two
files parses once per parser), and cycles are detected with a clear
error naming the chain. Import paths resolve relative to the
importing file's directory, so schema libraries can be shared across
projects by path.

Imports are a file-level concern handled before tokenization: the
engine language itself does not change.
"""

# Import Modules
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

# One import line: import "relative/path.nastruct";
_IMPORT_RE: re.Pattern[str] = re.compile(
    r'^[ \t]*import[ \t]+"([^"]+)"[ \t]*;[ \t]*$',
    re.MULTILINE,
)


def strip_imports(content: str) -> str:
    """
    Removes import declarations from schema source.

    Args:
        content (str): Raw .nastruct source.

    Returns:
        str: Source with import lines removed.
    """

    return _IMPORT_RE.sub("", content)


def extract_imports(content: str) -> list[str]:
    """
    Returns the import paths declared in a schema file, in order.

    Args:
        content (str): Raw .nastruct source.

    Returns:
        list[str]: Raw path strings as written.
    """

    return _IMPORT_RE.findall(content)


def resolve_schema_entry(
    entry: Path,
    seen: Optional[set[Path]] = None,
) -> list[tuple[Path, str]]:
    """
    Resolves a schema file and its transitive imports to parse order.

    Imported files come before their importer; every file appears at
    most once (first encounter wins). The entry file is always
    processed, even when already in `seen` (callers seed `seen` with
    files they parse elsewhere). The returned contents have their
    import lines stripped, ready for the parser.

    Args:
        entry (Path): The schema file to load.
        seen (Optional[set[Path]]): Canonical paths already resolved
            (shared across a multi-entry parse to deduplicate
            libraries). Created when omitted.

    Returns:
        list[tuple[Path, str]]: (absolute path, stripped content) in
            dependency order.

    Raises:
        FileNotFoundError: If an imported file does not exist.
        ValueError: If imports form a cycle.
    """

    visited: set[Path] = seen if seen is not None else set()
    stack: list[Path] = []

    return _resolve_file(
        path=entry, visited=visited, stack=stack, is_entry=True
    )


def _resolve_file(
    path: Path,
    visited: set[Path],
    stack: list[Path],
    is_entry: bool = False,
) -> list[tuple[Path, str]]:
    """
    Recursively resolves one schema file and its imports.

    Args:
        path (Path): Schema file to resolve.
        visited (set[Path]): Canonical paths already resolved.
        stack (list[Path]): In-progress chain, for cycle errors.
        is_entry (bool): True for a top-level entry, which is always
            processed regardless of `visited`.

    Returns:
        list[tuple[Path, str]]: Resolved files in dependency order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If imports form a cycle.
    """

    canonical: Path = path.resolve()

    if canonical in stack:
        chain: str = " -> ".join(
            str(item.name) for item in stack[stack.index(canonical):]
        )

        raise ValueError(
            f"Circular schema import: {chain} -> {canonical.name}"
        )

    if canonical in visited and not is_entry:
        return []

    if not canonical.is_file():
        raise FileNotFoundError(
            f"Imported schema file not found: '{path}'"
            + (
                f" (imported from '{stack[-1]}')"
                if stack
                else ""
            )
        )

    content: str = canonical.read_text(encoding="utf-8")

    result: list[tuple[Path, str]] = []
    stack.append(canonical)

    try:

        for raw_import in extract_imports(content=content):
            imported: Path = (
                canonical.parent / raw_import
            ).resolve()

            result.extend(
                _resolve_file(path=imported, visited=visited, stack=stack)
            )

    finally:
        stack.pop()

    visited.add(canonical)

    result.append((canonical, strip_imports(content=content)))

    return result
