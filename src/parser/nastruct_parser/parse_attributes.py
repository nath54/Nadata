"""
TODO
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.parser.utils.brackets import BracketTree
from src.parser.nastruct_parser.tmp_structs import TmpAttribute
from src.parser.nastruct_parser.struct_body_state import StructBodyScanState
from src.parser.nastruct_parser.nastruct_parser_states import NaStructParserStates


def parse_struct_attribute_case(
    parser_state: NaStructParserStates,
    scan_state: StructBodyScanState
) -> None:
    """
    Detects and dispatches a single attribute declaration of a struct body.

    Args:
        state (StructBodyScanState): Mutable scanning state (updated in place).
    """

    # Find the closing ; within the struct body only
    semicolon_index: int = scan_state.content.find(";", scan_state.cursor, scan_state.struct_end)

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
    parser_state: NaStructParserStates,
    struct_name: str,
    attr_start: int,
    attr_end: int,
    content: str,
    bracket_inside: Optional[BracketTree]
) -> None:
    """
    Parses a single struct attribute declaration.

    Args:
        struct_name (str): Name of the struct owning the attribute.
        attr_start (int): Index of the first character of the declaration.
        attr_end (int): Index of the closing ';' of the declaration.
        content (str): Text of the file.
        bracket_inside (Optional[BracketTree]): Bracket tree contained in the
            declaration (e.g. '{} ' or '[]' default value), None if none.
    """

    pass
