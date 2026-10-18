"""
Declaration text parsing shared by attributes and callable parameters.

Provides the parse_declaration_text helper extracting a (name, type, default)
triplet from a raw 'name: Type = default' declaration text. The type and the
default value are returned as raw strings, to be resolved in a later phase.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.parser.expressions.splitter import find_top_level
from src.parser.expressions.tokenizer import Token, TokenType, Tokenizer


def parse_declaration_text(
    declaration_text: str
) -> tuple[str, str, Optional[str]]:
    """
    Parses a 'name: Type = default' declaration text into its components.

    The split points are token-based, so '=' inside comparisons ('==', '<=',
    '>=', '!=', '->') and ':' inside dict literals are never mistaken for
    declaration separators.

    Args:
        declaration_text (str): Raw declaration text without its trailing ';'.

    Returns:
        tuple[str, str, Optional[str]]: The declaration name, the raw type
            annotation string, and the raw default value expression string
            (None if the declaration has no default value).

    Raises:
        SyntaxError: If the declaration text is malformed.
    """

    text: str = declaration_text.strip()

    # Check the declaration text is present
    if not text:
        raise SyntaxError("Empty declaration")

    tokens: list[Token] = Tokenizer(content=text).tokenize()

    # Locate the default value separator: the first top-level '=' token
    equals_token_index: int = find_top_level(tokens=tokens, value="=")

    default_value: Optional[str] = None
    type_end: int = len(text)

    # Split off the default value if present
    if equals_token_index != -1:
        equals_token: Token = tokens[equals_token_index]

        default_value = text[equals_token.end_position:].strip()
        type_end = equals_token.position

        # Check a default value is not empty
        if not default_value:
            raise SyntaxError(f"Empty default value in declaration: '{declaration_text}'")

        tokens = tokens[:equals_token_index]

    # Locate the name / type separator: the first top-level ':' token
    colon_token_index: int = find_top_level(tokens=tokens, value=":")

    # Check the name/type separator is present
    if colon_token_index == -1:
        raise SyntaxError(f"Missing ':' type separator in declaration: '{declaration_text}'")

    colon_token: Token = tokens[colon_token_index]

    declaration_name: str = text[:colon_token.position].strip()
    declaration_type: str = text[colon_token.end_position:type_end].strip()

    # Check the name is present
    if not declaration_name:
        raise SyntaxError(f"Missing name in declaration: '{declaration_text}'")

    # Check the type is present
    if not declaration_type:
        raise SyntaxError(f"Missing type in declaration: '{declaration_text}'")

    # Check the name is a single token
    if len(declaration_name.split()) != 1:
        raise SyntaxError(f"Malformed declaration name: '{declaration_text}'")

    return declaration_name, declaration_type, default_value
