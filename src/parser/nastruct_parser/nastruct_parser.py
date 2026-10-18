"""
Parser module for .nastruct schema files.
"""

# Import Modules
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path

from src.core.struct.struct import Struct
from src.core.database.database import NaDataDatabase
from src.core.struct.struct_method import StructMethod
from src.parser.utils.comments_remove import CommentsRemover
from src.parser.utils.brackets import BracketsAndStrings, BracketTree


@dataclass
class TmpAttribute:
    """
    TODO
    """

    attribute_name: str
    attribute_type: str
    attribute_default_value: Optional[Any] = None  # TODO: should we already parse the expressions in here?


@dataclass
class TmpFunction:
    """
    TODO
    """

    fn_name: str
    fn_argument: dict[str, TmpAttribute] = field(default_factory=dict)
    fn_output_type: str

    fn_body: list[Any] = field(default_factory=list)  # TODO: should we already parse the expressions in here?


@dataclass
class TmpStruct:
    """
    TODO
    """

    struct_name: str
    struct_parent: Optional[str] = None

    attributes: dict[str, TmpAttribute] = field(default_factory=dict)
    methods: dict[str, TmpFunction] = field(default_factory=dict)


@dataclass
class StructBodyScanState:
    """
    Mutable scanning state while parsing a struct body.

    Holds all the context of the struct being scanned: the case dispatcher
    methods read and update it, while the item-level parsers receive explicit
    per-item arguments extracted from it.

    Attributes:
        struct_name (str): Full header text of the struct being scanned.
        struct_end (int): Index of the closing brace of the struct body.
        content (str): Text of the file being parsed.
        cursor (int): Current character index in the content.
        brackets_inside (list[BracketTree]): Bracket trees of the struct body
            not yet consumed.
        next_bracket (Optional[BracketTree]): Next bracket tree to be consumed.
    """

    struct_name: str
    struct_end: int
    content: str
    cursor: int
    brackets_inside: list[BracketTree] = field(default_factory=list)
    next_bracket: Optional[BracketTree] = None

    def pop_next_bracket(self) -> Optional[BracketTree]:
        """
        Pops the next bracket tree from the waiting list.

        Returns:
            Optional[BracketTree]: Next bracket tree, or None if exhausted.
        """

        if not self.brackets_inside:
            return None

        return self.brackets_inside.pop(0)


class NaStructParser:
    """
    Parser for .nastruct schema files.
    """

    def __init__(
        self,
        database: Optional[NaDataDatabase] = None
    ) -> None:
        """
        Init function.
        """

        # Database where we will register the structs and functions
        self.database: NaDataDatabase = (
            database if database is not None else NaDataDatabase()
        )

        # Keys are dependencies we still are waiting for, and values the list of items needing it
        self.unresolved_dependencies: dict[str, list[str]] = {}

        self.waiting_structs: dict[str, Struct] = {}
        self.waiting_structs_deps: dict[str, set[tuple[str, str]]] = {}
        # Key is struct name, value is list of (dep name, dep type (struct or fn))

        self.waiting_fns: dict[str, StructMethod] = {}
        self.waiting_fns_deps: dict[str, set[tuple[str, str]]] = {}

        self.tmp_struct: dict[str, TmpStruct] = {}
        self.tmp_functions: dict[str, TmpFunction] = {}


    def parse_folder(self, folder_path: str) -> NaDataDatabase:
        """
        Parses all .nastruct files found in the specified folder into the database.

        Args:
            folder_path (str): Path to directory containing .nastruct files.

        Returns:
            NaDataDatabase: Populated database instance.
        """

        # Locate all .nastruct files in folder
        all_structs_files: list[str] = self.find_all_nastruct_files(
            folder_path=folder_path
        )

        # Parse each file
        for file_path in all_structs_files:
            with open(file_path, "r", encoding="utf-8") as file:
                self.parse_file(content=file.read())

        return self.database


    def find_all_nastruct_files(self, folder_path: str) -> list[str]:
        """
        Finds all .nastruct files recursively in a folder.

        Args:
            folder_path (str): Root folder path to search.

        Returns:
            list[str]: List of absolute file paths.
        """

        return [
            str(file.absolute())
            for file in Path(folder_path).rglob("*.nastruct")
        ]


    def parse_file(self, content: str) -> None:
        """
        Parses source content block by block for classes and standalone functions.

        Args:
            content (str): Full text content of the .nastruct file.

        Raises:
            SyntaxError: If a block header or matching closing brace is not found.
        """

        # Remove all the comments present in the file to avoid issues with the rest of the parsing
        content = CommentsRemover.remove_all_comments(content=content)

        # Find all the brackets and string couples
        bracket_trees: list[BracketTree]
        bracket_trees, _strings_couples = BracketsAndStrings.process(content=content)

        # Cursor tracking the end of the last processed block
        cursor: int = 0

        bt: BracketTree

        # For each top-level bracket tree, in file order
        for bt in bracket_trees:

            # Get the text between the previous block and this opening bracket
            header_text: str = content[cursor:bt.open_index].strip()

            # Extract the block keyword (first token of the header)
            keyword: str = header_text.split(maxsplit=1)[0] if header_text else ""

            # First case: class
            if keyword == "class":

                # Extract struct name and parent
                struct_name: str = ""
                struct_parent: Optional[str] = None

                self.parse_struct(
                    struct_name=struct_name,
                    struct_parent=struct_parent,
                    struct_start=bt.open_index + 1,
                    struct_end=bt.close_index,
                    brackets_inside=bt.inside,
                    content=content
                )

            # Second case: fn
            elif keyword == "fn":
                self.parse_fn(
                    name=header_text,
                    fn_start=bt.open_index + 1,
                    fn_end=bt.close_index,
                    brackets_inside=bt.inside,
                    content=content
                )

            # Syntax error
            else:
                raise SyntaxError(
                    f"Expected 'class' or 'fn' block before index {bt.open_index}, "
                    f"found: '{header_text[:40]}'"
                )

            # Continue after the closing bracket for the next block
            cursor = bt.close_index + 1


    def parse_struct(
        self,
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

        # Check if struct name already exists
        if struct_name in self.waiting_structs:  # TODO: also check in self.database
            raise SyntaxError("TODO")

        # Prepare the tmp struct
        self.tmp_struct[struct_name] = TmpStruct(
            struct_name=struct_name,
            struct_parent=struct_parent
        )

        # Initialize the mutable scanning state of the struct body
        # (struct_start is only used to initialize the cursor)
        state: StructBodyScanState = StructBodyScanState(
            struct_name=struct_name,
            struct_end=struct_end,
            content=content,
            cursor=struct_start,
            brackets_inside=brackets_inside
        )
        state.next_bracket = state.pop_next_bracket()

        # While we have something in the struct brackets
        while state.cursor < state.struct_end:

            # Manage white spaces (spaces, tabs, newlines, and blanked comments)
            if state.content[state.cursor].isspace():
                state.cursor += 1
                continue

            # Class method declaration
            if state.content[state.cursor] == "@":
                self.parse_struct_method_case(state=state)

            # Else, it is an attribut definition
            else:
                self.parse_struct_attribute_case(state=state)


    def parse_struct_attribute_case(
        self,
        state: StructBodyScanState
    ) -> None:
        """
        Detects and dispatches a single attribute declaration of a struct body.

        Args:
            state (StructBodyScanState): Mutable scanning state (updated in place).
        """

        # Find the closing ; within the struct body only
        semicolon_index: int = state.content.find(";", state.cursor, state.struct_end)

        # Check closing ;
        if semicolon_index == -1:
            raise SyntaxError(
                f"Missing closing ';' for attribute starting at index {state.cursor}"
            )

        bracket_inside: Optional[BracketTree] = None

        # Check next bracket
        if (
            state.next_bracket is not None
            and state.next_bracket.open_index < semicolon_index
        ):

            # Mark the bracket as inside the attribute declaration
            bracket_inside = state.next_bracket

            # Advance the next bracket
            state.next_bracket = state.pop_next_bracket()

        # Process the attribute
        self.parse_struct_attribute(
            struct_name=state.struct_name,
            attr_start=state.cursor,
            attr_end=semicolon_index,
            content=state.content,
            bracket_inside=bracket_inside
        )

        # Advance the cursor
        state.cursor = semicolon_index + 1


    def parse_struct_attribute(
        self,
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


    def parse_struct_method_case(
        self,
        state: StructBodyScanState
    ) -> None:
        """
        Detects and dispatches a single method declaration of a struct body.

        Args:
            state (StructBodyScanState): Mutable scanning state (updated in place).
        """

        # Check bracket
        if state.next_bracket is None:
            raise SyntaxError(
                f"Method declaration at index {state.cursor} has no body '{{...}}'"
            )

        # Get text before (the full method header: name, params, return type)
        method_header: str = (
            state.content[state.cursor + 1:state.next_bracket.open_index].strip()
        )

        # Check the text is present
        if not method_header:
            raise SyntaxError(f"Empty method declaration at index {state.cursor}")

        # A method declaration requires a parameter list '(...)'
        paren_index: int = method_header.find("(")

        # If there are no parenthesis, it is a syntax error
        if paren_index == -1:
            raise SyntaxError(
                f"Method declaration at index {state.cursor} is missing its "
                f"parameter list '(...)': '@{method_header[:20]}...'"
            )

        # The method name is the text before the parameter list
        method_name: str = method_header[:paren_index].strip()

        # Check the method name is present
        if not method_name:
            raise SyntaxError(f"Missing method name at index {state.cursor}")

        # Parse the struct method
        self.parse_struct_method(
            struct_name=state.struct_name,
            struct_method_name=method_name,
            header=method_header,
            method_start=state.next_bracket.open_index,
            method_end=state.next_bracket.close_index,
            content=state.content,
            brackets_inside=state.next_bracket.inside
        )

        # Advance the cursor
        state.cursor = state.next_bracket.close_index + 1

        # Advance the next bracket
        state.next_bracket = state.pop_next_bracket()


    def parse_struct_method(
        self,
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
        self,
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
