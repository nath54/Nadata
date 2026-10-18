"""
Tokenizer for Nadata expressions and declarations.

Converts raw source text into a flat list of typed tokens, handling string
literals with escapes, numeric literals, identifiers, keywords, operators, and
punctuation. Used by the expression parser and the declaration splitters.
"""

# Import Modules
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TokenType(Enum):
    """
    Enumeration of the token types produced by the Tokenizer.
    """

    IDENTIFIER = "identifier"
    KEYWORD = "keyword"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    OPERATOR = "operator"
    PUNCTUATION = "punctuation"
    EOF = "eof"


@dataclass
class Token:
    """
    Single lexical token with its position in the source text.

    Attributes:
        token_type (TokenType): Category of the token.
        value (str): Decoded token value (escapes resolved for strings).
        position (int): Index of the first source character of the token.
        end_position (int): Index just after the last source character.
    """

    token_type: TokenType
    value: str
    position: int
    end_position: int


# Language keywords (logical operators are keyword-only per the specification)
KEYWORDS: frozenset[str] = frozenset({
    "class", "fn", "if", "else", "return",
    "true", "false", "and", "or", "xor", "nand",
})

# Two-character operators, checked before single-character ones
MULTI_CHAR_OPERATORS: tuple[str, ...] = ("==", "!=", "<=", ">=", "->")

# Single-character operator symbols
SINGLE_CHAR_OPERATORS: str = "+-*/%<>=!"

# Punctuation characters
PUNCTUATION_CHARS: str = ".,:;()[]{}"

# Escape character used inside string literals
ESCAPE_CHAR: str = "\\"

# Supported escape sequence decodings inside string literals
ESCAPE_MAPPINGS: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "n": "\n",
    "t": "\t",
}


class Tokenizer:
    """
    Converts source text into a list of typed tokens in a single left-to-right
    scan. Comments are expected to have been removed beforehand.
    """

    def __init__(self, content: str) -> None:
        """
        Initializes the Tokenizer instance.

        Args:
            content (str): Comment-free source text to tokenize.
        """

        self.content: str = content
        self.cursor: int = 0
        self.token_start: int = 0

    def tokenize(self) -> list[Token]:
        """
        Tokenizes the whole content into a token list.

        Returns:
            list[Token]: Tokens in source order, terminated by an EOF token.

        Raises:
            SyntaxError: On unterminated strings, invalid escapes, or
                unexpected characters.
        """

        tokens: list[Token] = []

        # Scan until the end of the content
        while self.cursor < len(self.content):

            # Skip insignificant whitespace between tokens
            if self.content[self.cursor].isspace():
                self.cursor += 1
                continue

            previous_cursor: int = self.cursor

            tokens.append(self._scan_next_token())

            # Guard against non-advancing scans (would loop forever)
            if self.cursor <= previous_cursor:
                raise RuntimeError(
                    f"Tokenizer failed to advance at index {previous_cursor}"
                )

        # Terminate with an EOF token for the parser convenience
        tokens.append(Token(
            token_type=TokenType.EOF,
            value="",
            position=len(self.content),
            end_position=len(self.content)
        ))

        return tokens

    def _scan_next_token(self) -> Token:
        """
        Scans a single token starting at the current cursor.

        Returns:
            Token: The next token.

        Raises:
            SyntaxError: If no token can start at the cursor.
        """

        char: str = self.content[self.cursor]

        # Dispatch on the first character
        if char == "\"":
            return self._scan_string()

        if char.isdigit():
            return self._scan_number()

        if char.isalpha() or char == "_":
            return self._scan_identifier()

        if self.content[self.cursor:self.cursor + 2] in MULTI_CHAR_OPERATORS:
            return self._scan_multi_char_operator()

        if char in SINGLE_CHAR_OPERATORS:
            self.token_start = self.cursor
            self.cursor += 1

            return self._make_token(token_type=TokenType.OPERATOR, value=char)

        if char in PUNCTUATION_CHARS:
            self.token_start = self.cursor
            self.cursor += 1

            return self._make_token(token_type=TokenType.PUNCTUATION, value=char)

        raise SyntaxError(
            f"Unexpected character '{char}' at index {self.cursor}"
        )

    def _make_token(self, token_type: TokenType, value: Optional[str] = None) -> Token:
        """
        Builds a token spanning the characters consumed so far.

        Args:
            token_type (TokenType): Category of the token.
            value (Optional[str], optional): Decoded token value. Defaults to
                None (sliced from the source text).

        Returns:
            Token: The built token.
        """

        # Default value is the raw source slice (used by numbers, identifiers...)
        if value is None:
            value = self.content[self.token_start:self.cursor]

        token: Token = Token(
            token_type=token_type,
            value=value,
            position=self.token_start,
            end_position=self.cursor
        )

        return token

    def _scan_string(self) -> Token:
        """
        Scans a double quoted string literal with escape decoding.

        Returns:
            Token: The STRING token with the decoded value.

        Raises:
            SyntaxError: On unterminated strings or invalid escape sequences.
        """

        self.token_start = self.cursor
        self.cursor += 1

        decoded: list[str] = []

        # Consume characters until the closing quote
        while self.cursor < len(self.content) and self.content[self.cursor] != "\"":

            # Handle escape sequences
            if self.content[self.cursor] == ESCAPE_CHAR:
                decoded.append(self._consume_escape())
                continue

            # Raw newline inside a string is a syntax error
            if self.content[self.cursor] == "\n":
                raise SyntaxError(
                    f"Unterminated string literal starting at index {self.token_start}"
                )

            decoded.append(self.content[self.cursor])
            self.cursor += 1

        # Check the string is terminated
        if self.cursor >= len(self.content):
            raise SyntaxError(
                f"Unterminated string literal starting at index {self.token_start}"
            )

        # Consume the closing quote
        self.cursor += 1

        return self._make_token(token_type=TokenType.STRING, value="".join(decoded))

    def _consume_escape(self) -> str:
        """
        Consumes a backslash escape sequence at the cursor.

        Returns:
            str: The decoded character.

        Raises:
            SyntaxError: If the escape sequence is invalid.
        """

        # Check the escaped character exists
        if self.cursor + 1 >= len(self.content):
            raise SyntaxError(
                f"Unterminated escape sequence at index {self.cursor}"
            )

        escaped_char: str = self.content[self.cursor + 1]

        # Check the escape is supported
        if escaped_char not in ESCAPE_MAPPINGS:
            raise SyntaxError(
                f"Invalid escape sequence '\\{escaped_char}' at index {self.cursor}"
            )

        self.cursor += 2

        return ESCAPE_MAPPINGS[escaped_char]

    def _scan_number(self) -> Token:
        """
        Scans an integer or float numeric literal.

        Returns:
            Token: The INT or FLOAT token.

        Raises:
            SyntaxError: On malformed fractional parts.
        """

        self.token_start = self.cursor

        # Consume integer digits
        while self.cursor < len(self.content) and self.content[self.cursor].isdigit():
            self.cursor += 1

        # Check for a fractional part (a '.' followed by a digit)
        if (
            self.cursor + 1 < len(self.content)
            and self.content[self.cursor] == "."
            and self.content[self.cursor + 1].isdigit()
        ):
            self.cursor += 1

            # Consume fractional digits
            while self.cursor < len(self.content) and self.content[self.cursor].isdigit():
                self.cursor += 1

            return self._make_token(token_type=TokenType.FLOAT)

        # A '.' not followed by a digit belongs to a later postfix token
        # (e.g. member access on a float literal like '1.5.method()')
        return self._make_token(token_type=TokenType.INT)

    def _scan_identifier(self) -> Token:
        """
        Scans an identifier or keyword.

        Returns:
            Token: The IDENTIFIER or KEYWORD token.
        """

        self.token_start = self.cursor

        # Consume identifier characters
        while (
            self.cursor < len(self.content)
            and (self.content[self.cursor].isalnum() or self.content[self.cursor] == "_")
        ):
            self.cursor += 1

        value: str = self.content[self.token_start:self.cursor]

        # Distinguish keywords from plain identifiers
        token_type: TokenType = (
            TokenType.KEYWORD if value in KEYWORDS else TokenType.IDENTIFIER
        )

        return self._make_token(token_type=token_type, value=value)

    def _scan_multi_char_operator(self) -> Token:
        """
        Scans a two-character operator at the cursor.

        Returns:
            Token: The OPERATOR token.
        """

        self.token_start = self.cursor
        operator: str = self.content[self.cursor:self.cursor + 2]

        self.cursor += 2

        return self._make_token(token_type=TokenType.OPERATOR, value=operator)
