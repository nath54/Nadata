"""
Identifier extraction for Nadata dependency analysis.

Provides the IdentifierExtractor utility collecting all identifier names used
in a raw source fragment (type annotation, default value expression, or method
body), without parsing expressions. Used by the resolution phase to build the
dependency graph and validate that all referenced names exist.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

import re


class IdentifierExtractor:
    """
    Collects identifier names from raw Nadata source fragments.

    The extraction is syntax-light: it walks the text skipping string
    literals and comments, and returns every token matching the identifier
    grammar, minus reserved keywords. No expression parsing is performed.
    """

    # Identifier token pattern
    _IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    # Reserved keywords that are not dependencies
    _KEYWORDS: frozenset[str] = frozenset({
        "class", "fn", "if", "else", "return", "true", "false", "and", "or", "not",
    })

    # String and comment delimiters
    _QUOTE: str = '"'
    _TRIPLE_QUOTE: str = '"""'
    _HASH: str = "#"
    _BACKSLASH: str = "\\"
    _NEWLINE: str = "\n"

    @staticmethod
    def extract_identifiers(text: str) -> set[str]:
        """
        Extracts all identifier names used in the given source fragment.

        Args:
            text (str): Raw fragment (type annotation, default expression,
                method body, ...). Comments are allowed but typically absent.

        Returns:
            set[str]: Set of identifier names (keywords excluded).
        """

        identifiers: set[str] = set()

        cursor: int = 0
        length: int = len(text)

        # Single left-to-right scan skipping strings and comments
        while cursor < length:
            char: str = text[cursor]

            # Skip string literals (with escapes)
            if char == IdentifierExtractor._QUOTE:
                cursor += 1

                while cursor < length:
                    if text[cursor] == IdentifierExtractor._BACKSLASH:
                        cursor += 2
                        continue

                    if text[cursor] == IdentifierExtractor._QUOTE:
                        cursor += 1
                        break

                    cursor += 1

                continue

            # Skip line comments
            if char == IdentifierExtractor._HASH:
                while cursor < length and text[cursor] != IdentifierExtractor._NEWLINE:
                    cursor += 1

                continue

            # Skip block comments
            if text.startswith(IdentifierExtractor._TRIPLE_QUOTE, cursor):
                cursor += 3

                while cursor < length and not text.startswith(
                    IdentifierExtractor._TRIPLE_QUOTE, cursor
                ):
                    cursor += 1

                cursor += 3
                continue

            # Collect identifier tokens
            if char.isalpha() or char == "_":
                match: Optional[re.Match[str]] = (
                    IdentifierExtractor._IDENTIFIER_PATTERN.match(text, cursor)
                )

                # A match is guaranteed since the cursor is on a letter or underscore
                if match is None:
                    cursor += 1
                    continue

                identifier: str = match.group(0)

                if identifier not in IdentifierExtractor._KEYWORDS:
                    identifiers.add(identifier)

                cursor = match.end()
                continue

            cursor += 1

        return identifiers
