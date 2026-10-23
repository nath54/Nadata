"""
Tests for schema file imports: `import "path.nastruct";` declarations
resolve in dependency order, dedupe across entries, and detect cycles
and missing files.
"""

# Import Modules
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.api import Nadata
from src.parser.nastruct_parser.imports import (
    extract_imports,
    resolve_schema_entry,
    strip_imports,
)
from src.session import load_folder_session


@pytest.fixture
def tmp_root() -> Any:
    """
    Creates a temp directory cleaned up after the test.

    Yields:
        Any: The temp directory Path.
    """

    path: Any = Path(tempfile.mkdtemp())

    try:
        yield path

    finally:
        shutil.rmtree(path=path, ignore_errors=True)


def _write(folder: Any, rel: str, text: str) -> Path:
    """
    Writes a file (creating parents) and returns its path.

    Args:
        folder (Any): Root folder Path.
        rel (str): Relative posix path.
        text (str): File content.

    Returns:
        Path: The written file's path.
    """

    target: Any = folder / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

    return target


class TestResolver:
    """
    The file-level import resolver.
    """

    def test_dependency_order(self, tmp_root: str) -> None:
        """
        Imported files come before their importer.
        """

        _write(
            folder=tmp_root, rel="lib/base.nastruct",
            text="class Animal { name: Str; }",
        )
        entry: Path = _write(
            folder=tmp_root, rel="app/main.nastruct",
            text='import "../lib/base.nastruct";\nclass Dog(Animal) { }',
        )

        resolved: list[tuple[Path, str]] = resolve_schema_entry(entry=entry)

        assert [path.name for path, _ in resolved] == [
            "base.nastruct", "main.nastruct",
        ]

    def test_imports_stripped_from_content(self, tmp_root: str) -> None:
        """
        Import lines are removed before parsing.
        """

        content: str = 'import "a.nastruct";\nimport "b.nastruct";\nclass C { }'

        assert "import" not in strip_imports(content=content)
        assert extract_imports(content=content) == ["a.nastruct", "b.nastruct"]

    def test_dedupe_across_entries(self, tmp_root: str) -> None:
        """
        A library imported by two entries parses once per seen set.
        """

        lib: Path = _write(
            folder=tmp_root, rel="lib.nastruct", text="class Animal { }"
        )
        first: Path = _write(
            folder=tmp_root, rel="a.nastruct", text='import "lib.nastruct";\nclass A(Animal) { }'
        )
        second: Path = _write(
            folder=tmp_root, rel="b.nastruct", text='import "lib.nastruct";\nclass B(Animal) { }'
        )

        seen: set[Path] = set()

        resolve_schema_entry(entry=first, seen=seen)
        resolved_second: list[tuple[Path, str]] = resolve_schema_entry(
            entry=second, seen=seen
        )

        # Only the second entry itself: the lib was already seen
        assert [path.name for path, _ in resolved_second] == ["b.nastruct"]
        assert lib in seen

    def test_entry_always_processed_even_if_seen(
        self, tmp_root: str
    ) -> None:
        """
        Seeding seen with an entry's path does not skip the entry.
        """

        entry: Path = _write(
            folder=tmp_root, rel="main.nastruct", text="class M { }"
        )

        seen: set[Path] = {entry.resolve()}

        resolved: list[tuple[Path, str]] = resolve_schema_entry(
            entry=entry, seen=seen
        )

        assert [path.name for path, _ in resolved] == ["main.nastruct"]

    def test_missing_file(self, tmp_root: str) -> None:
        """
        A missing import raises FileNotFoundError naming the importer.
        """

        entry: Path = _write(
            folder=tmp_root, rel="main.nastruct",
            text='import "ghost.nastruct";\nclass M { }',
        )

        with pytest.raises(FileNotFoundError):
            resolve_schema_entry(entry=entry)

    def test_cycle_detected(self, tmp_root: str) -> None:
        """
        Circular imports raise ValueError with the chain.
        """

        _write(
            folder=tmp_root, rel="a.nastruct",
            text='import "b.nastruct";\nclass A { }',
        )
        entry: Path = _write(
            folder=tmp_root, rel="b.nastruct",
            text='import "a.nastruct";\nclass B { }',
        )

        with pytest.raises(ValueError, match="Circular schema import"):
            resolve_schema_entry(entry=entry)

    def test_deep_chain_order(self, tmp_root: str) -> None:
        """
        A -> B -> C resolves C, B, A.
        """

        _write(folder=tmp_root, rel="c.nastruct", text="class C { }")
        _write(
            folder=tmp_root, rel="b.nastruct",
            text='import "c.nastruct";\nclass B(C) { }',
        )
        entry: Path = _write(
            folder=tmp_root, rel="a.nastruct",
            text='import "b.nastruct";\nclass A(B) { }',
        )

        resolved: list[tuple[Path, str]] = resolve_schema_entry(entry=entry)

        assert [path.name for path, _ in resolved] == [
            "c.nastruct", "b.nastruct", "a.nastruct",
        ]


class TestEngineIntegration:
    """
    Imports work through the public load surfaces.
    """

    def test_single_file_load_resolves_tree(self, tmp_root: str) -> None:
        """
        Nadata.load on one file pulls in its transitive imports.
        """

        _write(
            folder=tmp_root, rel="lib/base.nastruct",
            text=(
                "class Animal {\n"
                "    name: Str;\n"
                "    @describe() -> Str { return self.name; }\n"
                "}\n"
            ),
        )
        entry: Path = _write(
            folder=tmp_root, rel="app/main.nastruct",
            text=(
                'import "../lib/base.nastruct";\n'
                "class Dog(Animal) { breed: Str = \"mutt\"; }\n"
            ),
        )

        app: Any = Nadata.load(path=str(entry))

        assert set(app.struct_names) == {"Animal", "Dog"}

        # Data flows through the imported hierarchy
        app.execute_data(
            'd = Dog(name = "Rex"); print(d.describe(), d.breed);'
        )

    def test_folder_load_dedupes(self, tmp_root: str) -> None:
        """
        Folder loading parses shared imports exactly once.
        """

        _write(
            folder=tmp_root, rel="lib/base.nastruct",
            text="class Animal { name: Str; }",
        )
        _write(
            folder=tmp_root, rel="app/types.nastruct",
            text='import "../lib/base.nastruct";\nclass Dog(Animal) { }',
        )
        _write(
            folder=tmp_root, rel="app/extra.nastruct",
            text='import "../lib/base.nastruct";\nclass Cat(Animal) { }',
        )

        app: Any = Nadata.load(path=str(tmp_root / "app"))

        assert set(app.struct_names) == {"Animal", "Dog", "Cat"}

    def test_session_with_external_library(self, tmp_root: str) -> None:
        """
        A session project imports a schema library outside its folder;
        cross-file inheritance resolves and seeds succeed.
        """

        _write(
            folder=tmp_root, rel="shared/base.nastruct",
            text="class Animal { name: Str; }",
        )
        _write(
            folder=tmp_root, rel="sess/schema.nastruct",
            text=(
                'import "../shared/base.nastruct";\n'
                "class Dog(Animal) { breed: Str = \"mutt\"; }\n"
            ),
        )
        _write(
            folder=tmp_root, rel="sess/seed.nadata",
            text='#init\nrex = Dog(nid = "rex", name = "Rex");\nflush();\n',
        )

        database: Any
        result: Any

        database, result = load_folder_session(
            folder_path=str(tmp_root / "sess")
        )

        assert result.ok()
        assert set(database.tables_by_struct_name) == {"Animal", "Dog"}

        rex: Any = database.get_by_nid(struct_name="Dog", nid="rex")

        assert rex is not None
        assert rex.get_attribute(name="name").get() == "Rex"
