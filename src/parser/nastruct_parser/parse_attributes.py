"""
Attribute declaration parsing for .nastruct schema files.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.core.expressions.expression import Expression
from src.parser.expressions.dependencies import collect_dependencies
from src.parser.expressions.type_parser import TmpType
from src.parser.utils.brackets import BracketsAndStrings, BracketTree
from src.parser.nastruct_parser.declaration import parse_declaration_text
from src.parser.nastruct_parser.tmp_structs import TmpAttribute, TmpStruct
from src.parser.nastruct_parser.struct_body_state import StructBodyScanState
from src.parser.nastruct_parser.parser_state import NaStructParserState


def parse_struct_attribute_case(
    parser_state: NaStructParserState,
    scan_state: StructBodyScanState
) -> None:
    """
    Detects and dispatches a single attribute declaration of a struct body.

    Args:
        parser_state (NaStructParserState): Shared parsing session state.
        scan_state (StructBodyScanState): Mutable scanning state (updated in place).
    """

    # Find the closing ; at depth zero within the struct body only
    # (skips ';' nested in call arguments, arrays, dicts, and strings)
    semicolon_index: int = BracketsAndStrings.find_unnested(
        content=scan_state.content,
        target_chars=";",
        start=scan_state.cursor,
        end=scan_state.struct_end
    )

    # Check closing ;
    if semicolon_index == -1:
        raise SyntaxError(
            f"Missing closing ';' for attribute starting at index {scan_state.cursor}"
        )

    bracket_inside: Optional[BracketTree] = None

    # Check next bracket
    if (
        scan_state.next_bracket is not None
        and scan_state.next_bracket.open_index < semicolon_index
    ):

        # Mark the bracket as inside the attribute declaration
        bracket_inside = scan_state.next_bracket

        # Advance the next bracket
        scan_state.next_bracket = scan_state.pop_next_bracket()

    # Process the attribute
    parse_struct_attribute(
        parser_state=parser_state,
        struct_name=scan_state.struct_name,
        attr_start=scan_state.cursor,
        attr_end=semicolon_index,
        content=scan_state.content,
        bracket_inside=bracket_inside
    )

    # Advance the cursor
    scan_state.cursor = semicolon_index + 1


def parse_struct_attribute(
    parser_state: NaStructParserState,
    struct_name: str,
    attr_start: int,
    attr_end: int,
    content: str,
    bracket_inside: Optional[BracketTree]
) -> None:
    """
    Parses a single struct attribute declaration into the tmp IR.

    Args:
        parser_state (NaStructParserState): Shared parsing session state.
        struct_name (str): Name of the struct owning the attribute.
        attr_start (int): Index of the first character of the declaration.
        attr_end (int): Index of the closing ';' of the declaration.
        content (str): Text of the file.
        bracket_inside (Optional[BracketTree]): Bracket tree contained in the
            declaration (e.g. '{} ' or '[]' default value), None if none.
    """

    # Extract the declaration text between the start and the closing ';'
    attr_text: str = content[attr_start:attr_end]

    # Parse the declaration into name, type, and default value
    attr_name: str
    attr_type: TmpType
    attr_default: Optional[Expression]
    attr_name, attr_type, attr_default = parse_declaration_text(
        declaration_text=attr_text
    )

    # Check the tmp struct exists (it is created before scanning the body)
    tmp_struct: Optional[TmpStruct] = parser_state.tmp_structs.get(struct_name)

    if tmp_struct is None:
        raise SyntaxError(
            f"Attribute '{attr_name}' refers to unknown struct '{struct_name}'"
        )

    # Check the attribute is not already declared on the struct
    if attr_name in tmp_struct.attributes:
        raise SyntaxError(
            f"Attribute '{attr_name}' is already defined on struct '{struct_name}'"
        )

    # Collect the dependencies from the type tree and the default expression
    dependencies: set[str] = attr_type.collect_names()

    if attr_default is not None:
        dependencies |= collect_dependencies(expression=attr_default)

    # Register the attribute in the tmp struct
    tmp_struct.attributes[attr_name] = TmpAttribute(
        attribute_name=attr_name,
        attribute_type=attr_type,
        attribute_default_value=attr_default,
        dependencies=dependencies
    )
