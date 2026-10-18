"""
Declaration text parsing shared by attributes and callable parameters.

Provides the parse_declaration_text helper extracting a (name, type, default)
triplet from a raw 'name: Type = default' declaration text. The type and the
default value are returned as raw strings, to be resolved in a later phase.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.parser.utils.brackets import BracketsAndStrings


def parse_declaration_text(
    declaration_text: str
) -> tuple[str, str, Optional[str]]:
    """
    Parses a 'name: Type = default' declaration text into its components.

    The default value detection and the name/type split are depth-aware:
    '=' and ':' characters nested in strings or brackets are ignored.

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

    # Detect the default value: first '=' at depth zero
    equals_index: int = BracketsAndStrings.find_unnested(
        content=text,
        target_chars="=",
        start=0,
        end=len(text)
    )

    default_value: Optional[str] = None

    # Split off the default value if present
    if equals_index != -1:
        default_value = text[equals_index + 1:].strip()
        text = text[:equals_index].strip()

    # Check a default value is not empty
    if default_value is not None and not default_value:
        raise SyntaxError(f"Empty default value in declaration: '{declaration_text}'")

    # Split the name and the type: first ':' at depth zero
    colon_index: int = BracketsAndStrings.find_unnested(
        content=text,
        target_chars=":",
        start=0,
        end=len(text)
    )

    # Check the name/type separator is present
    if colon_index == -1:
        raise SyntaxError(f"Missing ':' type separator in declaration: '{declaration_text}'")

    declaration_name: str = text[:colon_index].strip()
    declaration_type: str = text[colon_index + 1:].strip()

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
