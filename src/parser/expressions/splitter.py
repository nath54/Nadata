"""
Depth-aware token splitting utilities.

Provides helpers to locate or split token streams on separators appearing at
bracket depth zero, used for declaration parsing and parameter list parsing.
"""

# Import Modules
from __future__ import annotations

from src.parser.expressions.tokenizer import Token, TokenType


def find_top_level(
    tokens: list[Token],
    value: str
) -> int:
    """
    Finds the index of the first token with the given value at bracket depth zero.

    Args:
        tokens (list[Token]): Token stream to search (terminated by EOF).
        value (str): Token value to search for (e.g. '=' or ':').

    Returns:
        int: Index of the first matching top-level token, or -1 if none.
    """

    depth: int = 0

    # Single left-to-right scan tracking bracket depth
    for index, token in enumerate(tokens):

        # Stop at the EOF sentinel
        if token.token_type == TokenType.EOF:
            break

        # Track bracket nesting
        if token.value in "([{":
            depth += 1

        elif token.value in ")]}":
            depth -= 1

        # Separator found at depth zero
        elif depth == 0 and token.value == value:
            return index

    return -1


def split_top_level(
    tokens: list[Token],
    separator: str
) -> list[list[Token]]:
    """
    Splits a token stream on separators appearing at bracket depth zero.

    Args:
        tokens (list[Token]): Token stream to split (terminated by EOF).
        separator (str): Separator token value (e.g. ',').

    Returns:
        list[list[Token]]: Token segments between separators, in order. The
            EOF token is excluded; a trailing separator yields a final
            empty segment.
    """

    segments: list[list[Token]] = []

    current: list[Token] = []
    depth: int = 0

    # Single left-to-right scan tracking bracket depth
    for token in tokens:

        # Stop at the EOF sentinel
        if token.token_type == TokenType.EOF:
            break

        # Track bracket nesting
        if token.value in "([{":
            depth += 1

        elif token.value in ")]}":
            depth -= 1

        # Separator found at depth zero: close the current segment
        if depth == 0 and token.value == separator:
            segments.append(current)
            current = []
            continue

        current.append(token)

    segments.append(current)

    return segments
