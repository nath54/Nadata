"""
Callable definition parsing for .nastruct schema files.

Provides the parsers for struct method declarations and standalone function
declarations, which share the same 'name(params) -> ReturnType' header grammar.
"""

# Import Modules
from __future__ import annotations

from src.parser.utils.brackets import BracketTree
from src.parser.nastruct_parser.tmp_structs import TmpFunction
from src.parser.nastruct_parser.parser_state import NaStructParserState


def parse_struct_method(
    parser_state: NaStructParserState,
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
        parser_state (NaStructParserState): Shared parsing session state.
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
    parser_state: NaStructParserState,
    name: str,
    fn_start: int,
    fn_end: int,
    brackets_inside: list[BracketTree],
    content: str
) -> None:
    """
    Parses a standalone function definition block.

    Args:
        parser_state (NaStructParserState): Shared parsing session state.
        name (str): Full header text of the function
            (e.g. 'fn is_adult(h: Human) -> Bool').
        fn_start (int): Index of the first content character inside the braces.
        fn_end (int): Index of the closing brace of the block.
        brackets_inside (list[BracketTree]): Nested bracket trees inside the block.
        content (str): Text of the file.
    """

    pass
