"""
TODO
"""

# Import Modules
from __future__ import annotations

from src.parser.utils.brackets import BracketTree
from src.parser.nastruct_parser.tmp_structs import TmpFunction, TmpAttribute
from src.parser.nastruct_parser.struct_body_state import StructBodyScanState
from src.parser.nastruct_parser.nastruct_parser_states import NaStructParserStates





def parse_struct_method_case(
    parser_state: NaStructParserStates,
    scan_state: StructBodyScanState
) -> None:
    """
    Detects and dispatches a single method declaration of a struct body.

    Args:
        state (StructBodyScanState): Mutable scanning state (updated in place).
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
        method_start=scan_state.next_bracket.open_index,
        method_end=scan_state.next_bracket.close_index,
        content=scan_state.content,
        brackets_inside=scan_state.next_bracket.inside
    )

    # Advance the cursor
    scan_state.cursor = scan_state.next_bracket.close_index + 1

    # Advance the next bracket
    scan_state.next_bracket = scan_state.pop_next_bracket()


def parse_struct_method(
    parser_state: NaStructParserStates,
    struct_name: str,
    struct_method_name: str,
    header: str,
    method_start: int,
    method_end: int,
    content: str,
    brackets_inside: list[BracketTree]
) -> None:
    """
    Parses a struct method definition block.

    Args:
        struct_name (str): Name of the struct owning the method.
        struct_method_name (str): Name of the method (without '@').
        header (str): Full header text between '@' and the opening brace
            (e.g. 'age(now: DateTime) -> DateTime').
        method_start (int): Index of the first content character inside the braces.
        method_end (int): Index of the closing brace of the method body.
        content (str): Text of the file.
        brackets_inside (list[BracketTree]): Nested bracket trees in the body.
    """

    pass


def parse_fn(
    parser_state: NaStructParserStates,
    name: str,
    fn_start: int,
    fn_end: int,
    brackets_inside: list[BracketTree],
    content: str
) -> None:
    """
    Parses a standalone function definition block.

    Args:
        name (str): Full header text of the function
            (e.g. 'fn is_adult(h: Human) -> Bool').
        fn_start (int): Index of the first content character inside the braces.
        fn_end (int): Index of the closing brace of the block.
        brackets_inside (list[BracketTree]): Nested bracket trees inside the block.
        content (str): Text of the file.
    """

    pass
