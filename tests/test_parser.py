"""
Tests for NaStructParser block parsing and helper tools.
"""

# Import Modules
from __future__ import annotations
import pytest

from src.parser.nastruct_parser.nastruct_parser import NaStructParser


def test_parser_block_extraction() -> None:
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

    recorded: list[tuple[str, str, str]] = []

    parser: NaStructParser = NaStructParser()

    # Monkeypatch dispatch to capture parsed blocks
    def mock_parse_struct(name: str, content: str) -> None:
        recorded.append(("struct", name, content.strip()))

    def mock_parse_fn(name: str, content: str) -> None:
        recorded.append(("fn", name, content.strip()))

    parser.parse_struct = mock_parse_struct  # type: ignore[method-assign]
    parser.parse_fn = mock_parse_fn          # type: ignore[method-assign]

    parser.parse_file(content=content)

    assert len(recorded) == 2
    assert recorded[0][0] == "struct"
    assert recorded[0][1] == "class Human"
    assert "name: Str" in recorded[0][2]
    assert recorded[1][0] == "fn"
    assert recorded[1][1] == "fn is_adult(h: Human) -> Bool"
    assert "return h.age >= 18" in recorded[1][2]


def test_parser_unclosed_bracket_error() -> None:
    """
    Tests that a SyntaxError is raised when a closing brace is missing.
    """

    content: str = "class Broken { name: Str"
    parser: NaStructParser = NaStructParser()

    with pytest.raises(SyntaxError, match="Missing closing bracket"):
        parser.parse_file(content=content)


def test_parser_unexpected_token_error() -> None:
    """
    Tests that a SyntaxError is raised when an unexpected token is encountered.
    """

    content: str = "unexpected_token something {"
    parser: NaStructParser = NaStructParser()

    with pytest.raises(SyntaxError, match="Expected 'class' or 'fn' block"):
        parser.parse_file(content=content)
