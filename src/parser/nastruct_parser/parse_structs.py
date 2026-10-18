"""
TODO
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.parser.utils.brackets import BracketTree
from src.parser.nastruct_parser.tmp_structs import TmpStruct
from src.parser.nastruct_parser.struct_body_state import StructBodyScanState
from src.parser.nastruct_parser.nastruct_parser_states import NaStructParserStates
from src.parser.nastruct_parser.parse_attributes import parse_struct_attribute_case
from src.parser.nastruct_parser.parse_functions import parse_struct_method_case


def parse_struct_header(
    header_text: str
) -> tuple[str, Optional[str]]:
    """
    Extracts the struct name and optional parent name from a class header.

    Args:
        header_text (str): Full header text (e.g. 'class Human(Entity)').

    Returns:
        tuple[str, Optional[str]]: The struct name and the parent name
            (None if the struct has no parent).

    Raises:
        SyntaxError: If the header is malformed.
    """

    # Strip the 'class' keyword
    header_body: str = header_text[len("class"):].strip()

    # Case 1: inheritance 'class Human(Entity)'
    if header_body.endswith(")"):
        paren_index: int = header_body.find("(")

        # Check the opening parenthesis is present and well placed
        if paren_index == -1:
            raise SyntaxError(f"Malformed class header: '{header_text}'")

        struct_name: str = header_body[:paren_index].strip()
        struct_parent: str = header_body[paren_index + 1:-1].strip()

        # Check both names are present
        if not struct_name or not struct_parent:
            raise SyntaxError(f"Malformed class header: '{header_text}'")

        return struct_name, struct_parent

    # Case 2: no inheritance 'class Human'
    if "(" in header_body or ")" in header_body:
        raise SyntaxError(f"Malformed class header: '{header_text}'")

    # Check the struct name is present
    if not header_body:
        raise SyntaxError(f"Missing struct name in header: '{header_text}'")

    # Check the struct name is a single token
    if len(header_body.split()) != 1:
        raise SyntaxError(f"Malformed class header: '{header_text}'")

    return header_body, None

def parse_struct(
    parser_state: NaStructParserStates,
    struct_name: str,
    struct_parent: Optional[str],
    struct_start: int,
    struct_end: int,
    brackets_inside: list[BracketTree],
    content: str
) -> None:
    """
    Parses a struct definition block.

    Args:
        struct_name (str): Name of the struct
        struct_parent (Optional[str]): If the struct inherits from a parent struct
        struct_start (int): Index of the first content character inside the braces.
        struct_end (int): Index of the closing brace of the block.
        brackets_inside (list[BracketTree]): Nested bracket trees inside the block.
        content (str): Text of the file.
    """

    # Check if struct name already exists (tmp and resolved namespaces)
    if (
        struct_name in parser_state.tmp_struct
        or struct_name in parser_state.database.tables_by_struct_name
    ):
        raise SyntaxError(f"Struct '{struct_name}' is already defined")

    # Prepare the tmp struct
    parser_state.tmp_struct[struct_name] = TmpStruct(
        struct_name=struct_name,
        struct_parent=struct_parent
    )

    # Initialize the mutable scanning state of the struct body
    # (struct_start is only used to initialize the cursor)
    scan_state: StructBodyScanState = StructBodyScanState(
        struct_name=struct_name,
        struct_end=struct_end,
        content=content,
        cursor=struct_start,
        brackets_inside=brackets_inside
    )
    scan_state.next_bracket = scan_state.pop_next_bracket()

    # While we have something in the struct brackets
    while scan_state.cursor < scan_state.struct_end:

        # Manage white spaces (spaces, tabs, newlines, and blanked comments)
        if scan_state.content[scan_state.cursor].isspace():
            scan_state.cursor += 1
            continue

        # Class method declaration
        if scan_state.content[scan_state.cursor] == "@":
            parse_struct_method_case(
                parser_state=parser_state,
                scan_state=scan_state
            )

        # Else, it is an attribut definition
        else:
            parse_struct_attribute_case(
                parser_state=parser_state,
                scan_state=scan_state
            )
