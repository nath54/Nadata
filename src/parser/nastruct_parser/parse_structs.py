"""
Struct definition parsing for .nastruct schema files.

Provides the struct header parsing, the struct body scanning loop, and the
case dispatchers detecting attribute and method declarations.
"""

# Import Modules
from __future__ import annotations
from typing import Optional

from src.parser.utils.brackets import BracketTree
from src.parser.nastruct_parser.tmp_structs import TmpStruct
from src.parser.nastruct_parser.struct_body_state import StructBodyScanState
from src.parser.nastruct_parser.parser_state import NaStructParserState
from src.parser.nastruct_parser.parse_attributes import parse_struct_attribute_case
from src.parser.nastruct_parser.parse_callables import parse_struct_method


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
    parser_state: NaStructParserState,
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
        parser_state (NaStructParserState): Shared parsing session state.
        struct_name (str): Name of the struct.
        struct_parent (Optional[str]): Name of the parent struct, if any.
        struct_start (int): Index of the first content character inside the braces.
        struct_end (int): Index of the closing brace of the block.
        brackets_inside (list[BracketTree]): Nested bracket trees inside the block.
        content (str): Text of the file.
    """

    # Check if struct name already exists (tmp and resolved namespaces).
    # In merge mode (session schema re-merge), re-declaring a struct that
    # already lives in the database is allowed and replaces its members
    if (
        struct_name in parser_state.tmp_structs
        or (
            not parser_state.merge
            and struct_name in parser_state.database.tables_by_struct_name
        )
    ):
        raise SyntaxError(f"Struct '{struct_name}' is already defined")

    # Prepare the tmp struct
    parser_state.tmp_structs[struct_name] = TmpStruct(
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


def parse_struct_method_case(
    parser_state: NaStructParserState,
    scan_state: StructBodyScanState
) -> None:
    """
    Detects and dispatches a single method declaration of a struct body.

    Args:
        parser_state (NaStructParserState): Shared parsing session state.
        scan_state (StructBodyScanState): Mutable scanning state (updated in place).
    """

    # Check bracket
    if scan_state.next_bracket is None:
        raise SyntaxError(
            f"Method declaration at index {scan_state.cursor} has no body '{{...}}'"
        )

    # Get text before (the full method header: name, params, return type)
    method_header: str = (
        scan_state.content[scan_state.cursor + 1:scan_state.next_bracket.open_index].strip()
    )

    # Check the text is present
    if not method_header:
        raise SyntaxError(f"Empty method declaration at index {scan_state.cursor}")

    # A method declaration requires a parameter list '(...)'
    paren_index: int = method_header.find("(")

    # If there are no parenthesis, it is a syntax error
    if paren_index == -1:
        raise SyntaxError(
            f"Method declaration at index {scan_state.cursor} is missing its "
            f"parameter list '(...)': '@{method_header[:20]}...'"
        )

    # The method name is the text before the parameter list
    method_name: str = method_header[:paren_index].strip()

    # Check the method name is present
    if not method_name:
        raise SyntaxError(f"Missing method name at index {scan_state.cursor}")

    # Parse the struct method
    parse_struct_method(
        parser_state=parser_state,
        struct_name=scan_state.struct_name,
        struct_method_name=method_name,
        header=method_header,
        method_start=scan_state.next_bracket.open_index + 1,
        method_end=scan_state.next_bracket.close_index,
        content=scan_state.content,
        brackets_inside=scan_state.next_bracket.inside
    )

    # Advance the cursor
    scan_state.cursor = scan_state.next_bracket.close_index + 1

    # Advance the next bracket
    scan_state.next_bracket = scan_state.pop_next_bracket()
