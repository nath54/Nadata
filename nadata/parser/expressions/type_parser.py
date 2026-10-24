"""
Type annotation parsing for Nadata.

Provides the TmpType intermediate representation (a type name with nested
children) and the TypeParser converting a raw annotation string into it.
Shared by the .nastruct and .nadata parsers so types are parsed once into
structural data instead of being carried as raw strings.
"""

# Import Modules
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from nadata.parser.expressions.tokenizer import Token, TokenType, Tokenizer


@dataclass
class TmpType:
    """
    Intermediate representation of a parsed type annotation.

    A simple type (e.g. 'Str') has no children; a generic type (e.g.
    'Array[Entity]') has its name plus one child per type argument.

    Attributes:
        name (str): Type constructor name (e.g. 'Str', 'Array', 'Dict').
        children (list[TmpType]): Type arguments for generic types.
    """

    name: str
    children: list[TmpType] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the type to a plain JSON serializable dict.

        Returns:
            dict[str, Any]: Dict with 'name' and recursive 'children'.
        """

        return {
            "name": self.name,
            "children": [child.to_dict() for child in self.children],
        }

    def collect_names(self) -> set[str]:
        """
        Collects every type constructor name of the tree.

        Returns:
            set[str]: This type name plus all nested children names.
        """

        names: set[str] = {self.name}

        for child in self.children:
            names |= child.collect_names()

        return names

    def __repr__(self) -> str:
        """
        Returns string representation of the TmpType.

        Returns:
            str: Annotation style representation (e.g. 'Array[Entity]').
        """

        if not self.children:
            return self.name

        children_text: str = ", ".join(repr(child) for child in self.children)

        return f"{self.name}[{children_text}]"


class TypeParser:
    """
    Parses a token stream into a TmpType tree.
    """

    def __init__(self, tokens: list[Token]) -> None:
        """
        Initializes the TypeParser instance.

        Args:
            tokens (list[Token]): Token stream to parse (terminated by EOF).
        """

        self.tokens: list[Token] = tokens
        self.cursor: int = 0

    @classmethod
    def parse_text(cls, text: str) -> TmpType:
        """
        Tokenizes and parses a raw type annotation in one step.

        Args:
            text (str): Raw type annotation (e.g. 'Array[Entity]').

        Returns:
            TmpType: Parsed type tree.

        Raises:
            SyntaxError: On malformed or empty annotations.
        """

        tokens: list[Token] = Tokenizer(content=text).tokenize()

        parsed_type: TmpType
        cursor: int
        parsed_type, cursor = cls(tokens=tokens).parse_type_at(cursor=0)

        # Check nothing remains after the type
        if tokens[cursor].token_type != TokenType.EOF:
            raise SyntaxError(
                f"Unexpected token '{tokens[cursor].value}' in type '{text}'"
            )

        return parsed_type

    def parse_type_at(self, cursor: int) -> tuple[TmpType, int]:
        """
        Parses a single type starting at the given token cursor.

        Args:
            cursor (int): Token index to start parsing from.

        Returns:
            tuple[TmpType, int]: The parsed type and the new cursor.

        Raises:
            SyntaxError: On malformed type annotations.
        """

        token: Token = self.tokens[cursor]

        # A dimension literal (e.g. the 3 in 'Tensor[Float32, 3, 4]') is a
        # valid type argument and is carried as its digit-string name
        if token.token_type == TokenType.INT:
            cursor += 1

            return TmpType(name=token.value), cursor

        # A type starts with its constructor name
        if token.token_type != TokenType.IDENTIFIER:
            raise SyntaxError(
                f"Expected a type name but found '{token.value}' "
                f"at index {token.position}"
            )

        name: str = token.value
        cursor += 1

        children: list[TmpType] = []

        # Parse the optional generic argument list '[...]'
        if (
            self.tokens[cursor].token_type == TokenType.PUNCTUATION
            and self.tokens[cursor].value == "["
        ):
            cursor += 1

            # Parse comma separated type arguments
            while True:
                child: TmpType
                child, cursor = self.parse_type_at(cursor=cursor)
                children.append(child)

                current: Token = self.tokens[cursor]

                # Continue on comma, close on bracket
                if current.token_type == TokenType.PUNCTUATION and current.value == ",":
                    cursor += 1
                    continue

                if current.token_type == TokenType.PUNCTUATION and current.value == "]":
                    cursor += 1
                    break

                raise SyntaxError(
                    f"Expected ',' or ']' in type arguments of '{name}' "
                    f"but found '{current.value}' at index {current.position}"
                )

        return TmpType(name=name, children=children), cursor
