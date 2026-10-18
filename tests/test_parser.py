"""
Tests for the NaStructParser block extraction, bracket scanning,
and comment removal utilities.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

import pytest

from src.parser.nastruct_parser.nastruct_parser import NaStructParser
from src.parser.utils import (
    BracketTree,
    BracketsAndStrings,
    CommentsRemover,
    IdentifierExtractor,
)



class TestCommentsRemover:
    """
    Tests for the CommentsRemover utility.
    """

    def test_remove_line_comments(self) -> None:
        """
        Tests that '#' comments are blanked out up to end of line.
        """

        content: str = 'class A { # trailing comment\n    x: Int32\n}\n# full line\nend'
        result: str = CommentsRemover.remove_all_comments(content=content)

        assert "trailing comment" not in result
        assert "full line" not in result
        assert "class A {" in result
        assert len(result) == len(content)

    def test_remove_block_comments(self) -> None:
        """
        Tests that '\"\"\"' block comments are removed, including multiline ones.
        """

        content: str = 'class A {\n"""a comment\nspanning lines"""\n    x: Int32\n}'
        result: str = CommentsRemover.remove_all_comments(content=content)

        assert "a comment" not in result
        assert "spanning lines" not in result
        assert "class A {" in result
        assert "x: Int32" in result

    def test_string_content_preserved(self) -> None:
        """
        Tests that comment markers inside strings are not removed.
        """

        content: str = 'x: Str = "keep # this"'
        result: str = CommentsRemover.remove_all_comments(content=content)

        assert 'x: Str = "keep # this"' == result

    def test_escaped_quotes_preserved(self) -> None:
        """
        Tests that escaped quotes inside strings do not confuse the scanner.
        """

        content: str = 'x: Str = "escaped \\" quote # not a comment"'
        result: str = CommentsRemover.remove_all_comments(content=content)

        assert result == content

    def test_string_cannot_contain_triple_quote(self) -> None:
        """
        Tests that ' triple quote immediately after a string leads to an error
        (greedy left-to-right rule documented in SYNTAX.md).
        """

        with pytest.raises(SyntaxError, match="Unterminated string"):
            CommentsRemover.remove_all_comments(content='x: Str = "abc""" this"')

    def test_unterminated_constructs(self) -> None:
        """
        Tests that unterminated strings and block comments raise SyntaxError.
        """

        with pytest.raises(SyntaxError, match="Unterminated string"):
            CommentsRemover.remove_all_comments(content='x = "never closed')

        with pytest.raises(SyntaxError, match="Unterminated block comment"):
            CommentsRemover.remove_all_comments(content='x = """never closed')


class TestBracketsAndStrings:
    """
    Tests for the BracketsAndStrings scanner.
    """

    def test_nested_bracket_trees(self) -> None:
        """
        Tests that nested brackets produce a tree and top-level list is correct.
        """

        content: str = "fn f() { if (x) { return 1 } }"
        trees: list[BracketTree]
        trees, _ = BracketsAndStrings.process(content=content)

        assert len(trees) == 1
        assert content[trees[0].open_index] == "{"
        assert content[trees[0].close_index] == "}"
        assert len(trees[0].inside) == 1

    def test_brackets_inside_strings_ignored(self) -> None:
        """
        Tests that braces and quotes inside string literals are ignored.
        """

        content: str = 'x: Str = "} not a bracket {"'
        trees: list[BracketTree]
        trees, strings = BracketsAndStrings.process(content=content)

        assert trees == []
        assert list(strings.values()) == [len(content) - 1]

    def test_escaped_quotes_in_strings(self) -> None:
        """
        Tests that escaped quotes do not terminate the string.
        """

        content: str = 'x = "escaped \\" quote }"'
        trees: list[BracketTree]
        trees, _ = BracketsAndStrings.process(content=content)

        # The closing brace is inside the string, so no bracket tree exists
        assert trees == []

    def test_unbalanced_errors(self) -> None:
        """
        Tests that unbalanced brackets raise SyntaxError.
        """

        with pytest.raises(SyntaxError, match="Missing closing bracket"):
            BracketsAndStrings.process(content="class A { name: Str")

        with pytest.raises(SyntaxError, match="Unexpected closing bracket"):
            BracketsAndStrings.process(content="} oops {")


class TestNaStructParser:
    """
    Tests for the NaStructParser block dispatch.
    """

    def test_parser_block_extraction(self) -> None:
        """
        Tests that parse_file extracts classes and standalone functions.
        """

        content: str = """
        # File comment
        class Human {
            name: Str
            age: UInt8
        }

        # Standalone function
        fn is_adult(h: Human) -> Bool {
            return h.age >= 18
        }
        """

        recorded: list[tuple[str, str, Optional[str], int, int]] = []

        parser: NaStructParser = NaStructParser()

        # Monkeypatch dispatch to capture parsed blocks
        def mock_parse_struct(
            struct_name: str,
            struct_parent: Optional[str],
            struct_start: int,
            struct_end: int,
            brackets_inside: list[BracketTree],
            content: str
        ) -> None:
            recorded.append(("struct", struct_name, struct_parent, struct_start, struct_end))

        def mock_parse_fn(
            name: str,
            fn_start: int,
            fn_end: int,
            brackets_inside: list[BracketTree],
            content: str
        ) -> None:
            recorded.append(("fn", name, None, fn_start, fn_end))

        parser.parse_struct = mock_parse_struct  # type: ignore[method-assign]
        parser.parse_fn = mock_parse_fn          # type: ignore[method-assign]

        parser.parse_file(content=content)

        assert len(recorded) == 2
        assert recorded[0][0] == "struct"
        assert recorded[0][1] == "Human"
        assert recorded[0][2] is None
        assert recorded[1][0] == "fn"
        assert recorded[1][1] == "fn is_adult(h: Human) -> Bool"

    def test_parser_unclosed_bracket_error(self) -> None:
        """
        Tests that a SyntaxError is raised when a closing brace is missing.
        """

        content: str = "class Broken { name: Str"
        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="Missing closing bracket"):
            parser.parse_file(content=content)

    def test_parser_unexpected_token_error(self) -> None:
        """
        Tests that a SyntaxError is raised when an unexpected token is encountered.
        """

        content: str = "unexpected_token something { }"
        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="Expected 'class' or 'fn' block"):
            parser.parse_file(content=content)


class TestIdentifierExtractor:
    """
    Tests for the IdentifierExtractor dependency scanner.
    """

    def test_extract_from_method_body(self) -> None:
        """
        Tests that identifiers from calls, types, and declarations are collected.
        """

        body: str = (
            "DateTime age1 = h1.age(now); "
            "DateTime age2 = h2.age(now); "
            "return calculate_age_difference(h1; h2; now);"
        )

        identifiers: set[str] = IdentifierExtractor.extract_identifiers(text=body)

        assert "DateTime" in identifiers
        assert "age" in identifiers
        assert "now" in identifiers
        assert "calculate_age_difference" in identifiers
        assert "h1" in identifiers

    def test_keywords_and_strings_excluded(self) -> None:
        """
        Tests that keywords are filtered and string contents are ignored.
        """

        body: str = 'if (x == true and flag != "false and not a keyword") { return y; }'

        identifiers: set[str] = IdentifierExtractor.extract_identifiers(text=body)

        assert identifiers == {"x", "flag", "y"}
        assert "and" not in identifiers
        assert "not" not in identifiers

    def test_type_annotation_extraction(self) -> None:
        """
        Tests extraction from a raw type annotation string.
        """

        identifiers: set[str] = IdentifierExtractor.extract_identifiers(
            text="Array[Array[Float32]]"
        )

        assert identifiers == {"Array", "Float32"}
