"""
Tests for the NaStructParser block extraction, bracket scanning,
and comment removal utilities.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

import pytest

import src.parser.nastruct_parser.nastruct_parser as nastruct_parser_module
from src.core.expressions.expression import CallExpr, LiteralExpr
from src.parser.nastruct_parser.nastruct_parser import NaStructParser
from src.parser.nastruct_parser.parser_state import NaStructParserState
from src.parser.nastruct_parser.tmp_structs import TmpStruct
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
        (greedy left-to-right rule documented in docs/syntax.md).
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

    def test_parser_block_extraction(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            parser_state: NaStructParserState,
            struct_name: str,
            struct_parent: Optional[str],
            struct_start: int,
            struct_end: int,
            brackets_inside: list[BracketTree],
            content: str
        ) -> None:
            recorded.append(("struct", struct_name, struct_parent, struct_start, struct_end))

        def mock_parse_fn(
            parser_state: NaStructParserState,
            name: str,
            fn_start: int,
            fn_end: int,
            brackets_inside: list[BracketTree],
            content: str
        ) -> None:
            recorded.append(("fn", name, None, fn_start, fn_end))

        # Patch the module-level functions used by parse_file
        monkeypatch.setattr(nastruct_parser_module, "parse_struct", mock_parse_struct)
        monkeypatch.setattr(nastruct_parser_module, "parse_fn", mock_parse_fn)

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


class TestParseStructAttribute:
    """
    Tests for struct attribute declaration parsing into the tmp IR.
    """

    def test_attributes_populate_tmp_ir(self) -> None:
        """
        Tests that parsed attributes are registered with types and defaults.
        """

        parser: NaStructParser = NaStructParser()
        parser.parse_file(content=(
            "class Human {\n"
            "    name: Str;\n"
            "    gender: Str = \"?\";\n"
            "    size: UFloat16 = 1.75;\n"
            "}\n"
        ))

        tmp_human: TmpStruct = parser.parser_state.tmp_structs["Human"]

        assert list(tmp_human.attributes.keys()) == ["name", "gender", "size"]
        assert tmp_human.attributes["name"].attribute_type.name == "Str"
        assert tmp_human.attributes["name"].attribute_default_value is None

        # Defaults are stored as parsed Expression AST nodes
        gender_default = tmp_human.attributes["gender"].attribute_default_value

        assert isinstance(gender_default, LiteralExpr)
        assert gender_default.value.get() == "?"

        size_default = tmp_human.attributes["size"].attribute_default_value

        assert isinstance(size_default, LiteralExpr)
        assert size_default.value.get() == 1.75

        # Dependencies extracted from types and defaults
        assert tmp_human.attributes["name"].dependencies == {"Str"}
        assert tmp_human.attributes["size"].dependencies == {"UFloat16"}

    def test_separators_nested_in_calls_and_strings(self) -> None:
        """
        Tests that the attribute terminator ';' is depth-aware: commas inside
        nested calls and ';' inside strings do not terminate the attribute.
        """

        parser: NaStructParser = NaStructParser()
        parser.parse_file(content=(
            "class Human {\n"
            "    birthday: DateTime = DateTime(d=12, m=04, y=2003);\n"
            "    notes: Str = \"end; not yet = true: ok\";\n"
            "    tags: Array[Str] = [\"a; b\", \"c\"];\n"
            "}\n"
        ))

        tmp_human: TmpStruct = parser.parser_state.tmp_structs["Human"]

        # Defaults are stored as parsed Expression AST nodes
        birthday_default = tmp_human.attributes["birthday"].attribute_default_value

        assert isinstance(birthday_default, CallExpr)
        assert birthday_default.kwargs.keys() == {"d", "m", "y"}

        # Dependencies come from the parsed default expression AST: keyword
        # argument names ('d', 'm', 'y') are NOT symbol references.
        assert tmp_human.attributes["birthday"].dependencies == {"DateTime"}

        notes_default = tmp_human.attributes["notes"].attribute_default_value

        assert isinstance(notes_default, LiteralExpr)
        assert notes_default.value.get() == "end; not yet = true: ok"

        assert tmp_human.attributes["notes"].dependencies == {"Str"}
        assert tmp_human.attributes["tags"].attribute_type.to_dict() == {
            "name": "Array",
            "children": [{"name": "Str", "children": []}],
        }

    def test_duplicate_attribute_error(self) -> None:
        """
        Tests that declaring the same attribute twice raises a SyntaxError.
        """

        parser: NaStructParser = NaStructParser()

        with pytest.raises(SyntaxError, match="already defined"):
            parser.parse_file(content=(
                "class Human {\n"
                "    name: Str;\n"
                "    name: Str;\n"
                "}\n"
            ))
