"""
Default value expression parsing for .nastruct attributes.

Provides the parse_expression helper converting an attribute default value
source span into an Expression AST using the shared expression parser.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from nadata.core.expressions.expression import Expression
from nadata.parser.expressions.expression_parser import ExpressionParser
from nadata.parser.nastruct_parser.parser_state import NaStructParserState
from nadata.parser.utils.brackets import BracketTree


def parse_expression(
    parser_state: NaStructParserState,
    struct_name: str,
    attr_start: int,
    attr_end: int,
    content: str,
    bracket_inside: Optional[BracketTree]
) -> Expression:
    """
    Parses the default value expression of an attribute into an AST.

    Args:
        parser_state (NaStructParserState): Shared parsing session state
            (unused for now, kept for signature parity with the other
            attribute parsers).
        struct_name (str): Name of the struct owning the attribute
            (unused for now, kept for signature parity).
        attr_start (int): Index of the first character of the expression span.
        attr_end (int): Index just after the last character of the span.
        content (str): Text of the file.
        bracket_inside (Optional[BracketTree]): Bracket tree contained in the
            declaration (unused for now, kept for signature parity).

    Returns:
        Expression: Parsed expression AST.

    Raises:
        SyntaxError: If the expression span is empty or malformed.
    """

    # Extract the raw expression text between the span boundaries
    expression_text: str = content[attr_start:attr_end].strip()

    # Check the expression text is present
    if not expression_text:
        raise SyntaxError(
            f"Empty expression for struct '{struct_name}' "
            f"between indexes {attr_start} and {attr_end}"
        )

    # Delegate to the shared recursive descent expression parser
    return ExpressionParser.parse_text(text=expression_text)
